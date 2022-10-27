#include"../include/util/util.h"

int main(){
    char hash[32];
    const char input[2] = "a";
    calculate_hash(input,1,hash);
    for(int i=0;i<64;i++){
        printf("%c",nibble_from_bytes(hash,i));
    }
    return 0;
}