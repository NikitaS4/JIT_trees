#include "JITedTree.h"

// OS-dependent imports
#ifdef __linux__
    //linux
    #include <dlfcn.h>
#endif
// windows.h already included in JITedTree.h

#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cstdlib>

// init static members
const std::string JITedTree::dirPreffix = "JIT_files_model_";
size_t JITedTree::lastModel = 0;
// OS-dependent initializations
#ifdef __linux__
    // linux
    const std::string JITedTree::rmExe = "rm -r ";
    const std::string JITedTree::dexport = ""; // empty
    const std::string JITedTree::dirDelimeter = "/";

    #define CLOSE_LIBRARY(arg) {dlclose((arg));}
#elif _WIN32
    // windows
    const std::string JITedTree::rmExe = "rd /s /q ";
    const std::string JITedTree::dexport = "__declspec(dllexport)";
    const std::string JITedTree::dirDelimeter = "\\";

    #define CLOSE_LIBRARY(arg) {FreeLibrary((arg));}
#else // unsupported OS
    int a[-1]; // can not compile for unsupported OS
#endif


JITedTree::JITedTree(const size_t treeDepth, const size_t featureCnt): 
    treeDepth(treeDepth), innerNodes((1 << treeDepth) - 1), featureCnt(featureCnt),
    leafCnt(size_t(1) << treeDepth), dirName(dirPreffix + std::to_string(lastModel++)), treeCnt(0) {
    // create directory where sources and JIT compiled libs will be stored
    std::string makeDirCmd = "mkdir " + dirName;
    if (std::system(makeDirCmd.c_str()) != 0)
        throw std::runtime_error("JIT error: can't create directory");
}


void JITedTree::compileTree(const size_t* features, const FVal_t* thresholds,
    const Lab_t* leaves) {
    std::string fname = std::to_string(treeCnt);
    std::string fnameWithDir = dirName + dirDelimeter + fname;
    ++treeCnt;
    // create file with source code
    createSrc(fnameWithDir, features, thresholds, leaves);
    // compile to dynamically linked library
    if (compile(fname) != 0)
        throw std::runtime_error("JIT error: compile fault");
    // open compiled library
    openLib(fnameWithDir);
}


JITedTree::~JITedTree() {
    // close libs
    for (auto & curLib : libPtr) {
        if (curLib)
            CLOSE_LIBRARY(curLib);
    }
    // delete created sources & compiled libs
    std::string rmDirCmd = rmExe + dirName;
    int rc = std::system(rmDirCmd.c_str()); // rc unused - can't throw from destructor
}


Lab_t JITedTree::predictTree(const pytensor1& sample, const size_t treeNum) const {
    return treePredict[treeNum](sample.data());
}


Lab_t JITedTree::predictAllTrees(const pytensor1& sample) const {
    Lab_t cumSum = 0;
    const double* pData = sample.data();
    for (auto& curPredictor : treePredict)
        cumSum += curPredictor(pData);
    return cumSum;
}


Lab_t JITedTree::predictFromTo(const pytensor1& sample, const size_t from,
    const size_t to) const {
    Lab_t cumSum = 0;
    const double* pData = sample.data();
    for (size_t i = from; i < to; ++i)
        cumSum += treePredict[i](pData);
    return cumSum;
}


size_t JITedTree::getTreeCount() const {
    return treeCnt;
}


void JITedTree::popTree() {
    if (treeCnt > 0) {        
        CLOSE_LIBRARY(libPtr[treeCnt - 1]);
        treePredict.pop_back();
        libPtr.pop_back();
        --treeCnt;
    }
}


void JITedTree::createSrc(const std::string& fname, const size_t* features, 
    const FVal_t* thresholds, const Lab_t* leaves) {
    std::ofstream jitSrc(fname + ".cpp");
    jitSrc <<
        //"#include \"..\\..\\src\\common\\AtomicTypes.h\"\n"
        //"#include <cstddef>\n"
        "#ifdef __cplusplus\n"
        "extern \"C\" \n"
        "#endif\n"
        "double " << dexport << " predict(const double* sample) {\n"
        "static unsigned int features[] = {";
    for (size_t i = 0; i < treeDepth - 1; ++i) {
        jitSrc << features[i] << ", ";
    }
    jitSrc << features[treeDepth - 1] << "};"
        "static double thresholds[] = {";
    for (size_t i = 0; i < innerNodes - 1; ++i) {
        jitSrc << thresholds[i] << ", ";
    }
    jitSrc << thresholds[innerNodes - 1] << "};"
        "static double leaves[] = {";
    for (size_t i = 0; i < leafCnt - 1; ++i) {
        jitSrc << leaves[i] << ", ";
    }
    jitSrc << leaves[leafCnt - 1] << "};\n"
        "unsigned int curNode = 0;\n"
        "for (unsigned int h = 0; h < " << treeDepth << "; ++h) {\n"
        "if (sample[features[h]] < thresholds[curNode])\n"
        "   curNode = 2 * curNode + 1;\n"
        "else\n"
        "   curNode = 2 * curNode + 2;\n"
        "}\n"
        "return leaves[curNode - " << innerNodes << "];}\n";
    jitSrc.close();
}


int JITedTree::compile(const std::string& fname) {
    #ifdef __linux__
        // linux
        std::string fullFname = dirName + dirDelimeter + fname;
        std::string command = "g++ -shared -Wall " + fullFname + ".cpp -o " + fullFname + ".so";
        return std::system(command.c_str());
    #elif _WIN32
        // windows
        std::string command1 = "cd " + dirName + " & g++ -c " + fname + ".cpp";
        std::string command2 = "cd " + dirName + " & g++ -shared -o " + fname + ".dll " + fname + ".o -Wl,--out-implib," + fname + ".a";
        std::system(command1.c_str());
        return std::system(command2.c_str());
    #else // unsupported OS
        int a[-1]; // will not compile for unsupported OS
    #endif    
}


void JITedTree::openLib(const std::string& fname) {
    // OS-specific
    #ifdef __linux__
        // linux
        // open lib
        std::string dlFname = "./" + fname + ".so";
        void* pLib = dlopen(dlFname.c_str(), RTLD_LAZY);
        if (pLib == nullptr)
            throw std::runtime_error("JIT error: can't open .so");
        libPtr.push_back(pLib); // close lib on finish
        // load function
        Lab_t (*predJIT)(const double*) = 
            reinterpret_cast<Lab_t(*)(const double*)>(dlsym(pLib, "predict"));
    #elif _WIN32
        // windows
        // open lib
        std::string dlFname = ".\\" + fname + ".dll"; // get the name of the DLL
        HINSTANCE hDLL = LoadLibrary(dlFname.c_str());
        if (hDLL == NULL) {
            static const DWORD bitnessError = 193;
            if (bitnessError == GetLastError())
                throw std::runtime_error("JIT error: wrong bitness of JIT-compiled library, use MinGW for 64 bits");
            throw std::runtime_error("JIT error: can't open .dll with name: " + dlFname);
        }
        libPtr.push_back(hDLL); // to close lib on finish
        // load function
        Lab_t (*predJIT)(const double*) =
            (Lab_t(*)(const double*))GetProcAddress(hDLL, "predict");
    #else // unsupported OS
        int a[-1]; // can not compile for unsupported OS
    #endif
    // OS-independent
    if (predJIT == nullptr)
        throw std::runtime_error("JIT error: can't find proper function");
    treePredict.push_back(predJIT);
}
