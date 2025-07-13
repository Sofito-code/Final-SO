// matrix_multiply_fork_pipes.cpp – versión con estadísticas detalladas
// Compilar: g++ -O2 -std=c++17 matrix_multiply_fork_pipes.cpp -lcurl -o matmul

#include <bits/stdc++.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <unistd.h>
#include <nlohmann/json.hpp>     // https://github.com/nlohmann/json
#include <curl/curl.h>

using namespace std;
using namespace std::chrono;
using Matrix = vector<vector<float>>;

//---------------------------------- utilidades curl
size_t noop(char *, size_t s, size_t n, void *) { return s * n; }

bool sendJsonToFirebase(const nlohmann::json &payload, const string &url) {
    CURL *curl = curl_easy_init();
    if (!curl) { cerr << "[curl] init failed" << endl; return false; }
    string jsonStr = payload.dump();

    struct curl_slist *hdrs{nullptr};
    hdrs = curl_slist_append(hdrs, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonStr.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, noop); // ignorar respuesta

    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(hdrs);
    curl_easy_cleanup(curl);
    if (res != CURLE_OK) cerr << "[curl] " << curl_easy_strerror(res) << endl;
    return res == CURLE_OK;
}

//---------------------------------- matrices IO
Matrix readMatrix(const string &file, int &r, int &c) {
    ifstream f(file);
    if (!f.is_open()) { cerr << "No se pudo abrir " << file << endl; exit(1);}    
    Matrix M; string line; r = 0; c = 0;
    while (getline(f, line)) {
        istringstream iss(line); vector<float> row; float v;
        while (iss >> v) row.push_back(v);
        if (c == 0) c = row.size();
        else if ((int)row.size() != c) { cerr << "Filas con distinta longitud en " << file << endl; exit(1);}        
        M.push_back(move(row)); r++;
    }
    return M;
}

void writeMatrix(const string &file, const Matrix &M) {
    ofstream f(file);
    for (const auto &row: M) {
        for (size_t i=0;i<row.size();++i) f << row[i] << (i+1==row.size()?"\n":" ");
    }
}

//---------------------------------- multiplicación
void multSeq(const Matrix &A,const Matrix &B,Matrix &C){
    int n=A.size(), m=A[0].size(), p=B[0].size();
    for(int i=0;i<n;++i)
        for(int j=0;j<p;++j){
            float s=0; for(int k=0;k<m;++k) s+=A[i][k]*B[k][j];
            C[i][j]=s;
        }
}

void multForkPipes(const Matrix &A,const Matrix &B,Matrix &C,int proc){
    int n=A.size(), m=A[0].size(), p=B[0].size();
    vector<int> pipesR(proc), pipesW(proc);
    for(int id=0;id<proc;++id){
        int fd[2]; if(pipe(fd)==-1){perror("pipe"); exit(1);} pipesR[id]=fd[0]; pipesW[id]=fd[1];
        pid_t pid=fork();
        if(pid==0){ // hijo
            close(fd[0]);
            int s=(n/proc)*id+min(id,n%proc);
            int e=s+(n/proc)+(id<n%proc);
            vector<float> buf((e-s)*p);
            int idx=0;
            for(int i=s;i<e;++i)
                for(int j=0;j<p;++j){
                    float sum=0; for(int k=0;k<m;++k) sum+=A[i][k]*B[k][j];
                    buf[idx++]=sum;
                }
            write(fd[1],buf.data(),buf.size()*sizeof(float));
            close(fd[1]); _exit(0);
        } else {
            close(fd[1]);
        }
    }
    // padre lee
    for(int id=0;id<proc;++id){
        int s=(n/proc)*id+min(id,n%proc);
        int e=s+(n/proc)+(id<n%proc);
        vector<float> buf((e-s)*p);
        size_t total = buf.size() * sizeof(float);
        size_t read_bytes = 0;
        char* ptr = reinterpret_cast<char*>(buf.data());
        while (read_bytes < total) {
            ssize_t r = read(pipesR[id], ptr + read_bytes, total - read_bytes);
            if (r <= 0) { perror("read"); break; }
            read_bytes += r;
        }
        close(pipesR[id]);
        int idx=0; for(int i=s;i<e;++i) for(int j=0;j<p;++j) C[i][j]=buf[idx++];
    }
    while(wait(nullptr)>0); // esperar todos hijos
}

//---------------------------------- main
int main(int argc,char*argv[]){
    if(argc<4){
        cerr<<"Uso: "<<argv[0]<<" A.txt B.txt <procesos>"<<endl; return 1; }
    string fileA=argv[1], fileB=argv[2];
    int P=stoi(argv[3]);
    string fbURL = "https://[YOUR_DB].firebaseio.com/data.json";

    int rA,cA,rB,cB; Matrix A=readMatrix(fileA,rA,cA); Matrix B=readMatrix(fileB,rB,cB);
    if(cA!=rB){ cerr<<"Dimensiones incompatibles"<<endl; return 1; }
    Matrix Cseq(rA,vector<float>(cB)), Cpar(rA,vector<float>(cB));

    //------------------- secuencial
    auto t0=steady_clock::now(); multSeq(A,B,Cseq); auto t1=steady_clock::now();
    long long tSeqMs=duration_cast<milliseconds>(t1-t0).count();
    writeMatrix("C_seq.txt", Cseq);

    //------------------- paralela
    struct rusage beforePar{}, afterPar{};
    getrusage(RUSAGE_SELF,&beforePar);
    auto p0=steady_clock::now(); multForkPipes(A,B,Cpar,P); auto p1=steady_clock::now();
    getrusage(RUSAGE_SELF,&afterPar);
    long long tParMs=duration_cast<milliseconds>(p1-p0).count();
    writeMatrix("C_par.txt", Cpar);

    //------------------- métricas adicionales
    auto diffUser = (afterPar.ru_utime.tv_sec-beforePar.ru_utime.tv_sec)*1000 +
                    (afterPar.ru_utime.tv_usec-beforePar.ru_utime.tv_usec)/1000;
    auto diffSys  = (afterPar.ru_stime.tv_sec-beforePar.ru_stime.tv_sec)*1000 +
                    (afterPar.ru_stime.tv_usec-beforePar.ru_stime.tv_usec)/1000;
    long rssKB = afterPar.ru_maxrss; // pico en KB

    double speedup = (double)tSeqMs / tParMs;

    //------------------- enviar a Firebase
    nlohmann::json payload = {
        {"matrixRows", rA},
        {"matrixCols", cB},
        {"numProcesses", P},
        {"timeSeqMs", tSeqMs},
        {"timeParMs", tParMs},
        {"speedup", speedup},
        {"cpuUserMs", diffUser},
        {"cpuSysMs", diffSys},
        {"maxRssKB", rssKB},
        {"timestamp", duration_cast<seconds>(system_clock::now().time_since_epoch()).count()},
	    {"RB", "Ubuntu"}
    };

    curl_global_init(CURL_GLOBAL_DEFAULT);
    bool ok = sendJsonToFirebase(payload, fbURL);
    curl_global_cleanup();

    //------------------- salida local
    cout << fixed << setprecision(3);
    cout << "Secuencial: " << tSeqMs/1000.0 << " s\n";
    cout << "Paralelo("<<P<<"): " << tParMs/1000.0 << " s\n";
    cout << "Speedup: " << speedup << "x\n";
    cout << "CPU user(ms): "<< diffUser << "  sys(ms): "<< diffSys << "  pico RAM: "<< rssKB << " KB\n";
    if(ok) cout << "[Firebase] subida correcta" << endl; else cout << "[Firebase] fallo" << endl;

    // opcional: escribir matriz resultado
    //writeMatrix("C_par.txt", Cpar);
    return 0;
}
