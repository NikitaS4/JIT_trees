#include "TreeHolder.h"
#include "RegularTree.h"


TreeHolder::TreeHolder(const size_t treeDepth, const size_t featureCnt):
    treeDepth(treeDepth), innerNodes((1 << treeDepth) - 1), featureCnt(featureCnt),
    leafCnt(size_t(1) << treeDepth), treeCnt(0) {
    // ctor
}


TreeHolder::~TreeHolder() {
    // dtor
}


size_t TreeHolder::getTreeCount() const {
    return treeCnt;
}


TreeHolder* TreeHolder::createHolder(const size_t treeDepth,
    const size_t featureCnt) {
        return new RegularTree(treeDepth, featureCnt);
}
