# copy to statedb.go's directory
cp build/libgmpt.a go-ethereum/core/state
cp mpt-with-compress/lib/libgmpt.h go-ethereum/core/state
# copy to worker.go's directory
cp build/libgmpt.a go-ethereum/miner
cp mpt-with-compress/lib/libgmpt.h go-ethereum/miner
# copy to hasing.go's directory
cp build/libgmpt.a go-ethereum/core/types
cp mpt-with-compress/lib/libgmpt.h go-ethereum/core/types
# copy to test temp directory
cp build/libgmpt.a ~/code/temp
cp mpt-with-compress/lib/libgmpt.h ~/code/temp
# copy to blockchain.go's directory
cp build/libgmpt.a go-ethereum/core/
cp mpt-with-compress/lib/libgmpt.h go-ethereum/core/