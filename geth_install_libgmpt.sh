# copy to statedb.go's directory
cp build/libgmpt.a go-ethereum/core/state
cp mpt-with-compress/lib/libgmpt.h go-ethereum/core/state

# copy to test temp directory
cp build/libgmpt.a ~/code/temp
cp mpt-with-compress/lib/libgmpt.h ~/code/temp