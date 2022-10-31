#include<errno.h>
#include<sys/stat.h>
#include<string>
#include<stdio.h>

#include"rapidjson/document.h"
#include"rapidjson/reader.h"
#include "rapidjson/filereadstream.h"

#define MAX_FILE_SIZE 500000000
#define BITRATE       1024

static uint8_t  *h_data;
static uint64_t *d_data;
static uint8_t  *h_out;
static uint64_t *d_out;

int main(){
    // struct stat buf;
    // size_t size;
    // std::string filename;
    // if(stat(filename.c_str(), &buf) < 0) {
    //     fprintf(stderr, "stat %s failed: %s\n", filename, strerror(errno));
    //     return;
    // } 

    // if(buf.st_size == 0 || buf.st_size > MAX_FILE_SIZE) {
    //     fprintf(stderr, "%s wrong sized %d\n", filename, (int)buf.st_size);
    //     return;
    // }
    //                                     /* align the data on BITRATE/8 bytes */
    // size = ((buf.st_size-1)/(BITRATE/8) + 1)*(BITRATE/8);

    // FILE *in = fopen(filename.c_str(), "r");

    // if(in == NULL) {
    //     fprintf(stderr, "open %s failed: %s\n", filename, strerror(errno));
    //     return;
    // }

    // h_data = (uint8_t *)calloc(1, size);
    // //memset(h_data, 0x00, size);  
    // /* read in the document to be hashed */
    // if(fread(h_data, 1, (size_t)buf.st_size, in) < buf.st_size) {
    //     fprintf(stderr, "read %s failed: %s\n", filename, strerror(errno));
    //     return;
    // }

    // fclose(in);

    // return 0;
    std::FILE* fp = fopen("../dataset/transactions.json", "r");
    if (fp == 0)
    {
        printf("no file\n");
    }
    
    char readBuffer[200000];
    rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
    
    rapidjson::Document jsondata;
    jsondata.ParseStream(is);
    
    assert(jsondata.IsArray());
    for (size_t i = 0; i < jsondata.Size(); i++)
    {
        printf("%d\n",i);
        assert(jsondata[i].IsObject());
        printf("%d\n",jsondata[i].Size());
    }
    
    fclose(fp);
}
