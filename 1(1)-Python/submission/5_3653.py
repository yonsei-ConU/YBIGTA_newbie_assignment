from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Callable, List


"""
TODO:
- SegmentTree 구현하기
"""


T = TypeVar("T")
U = TypeVar("U")


class SegmentTree(Generic[T, U]):
    """ConU's non-recursive uniform segment tree implementation"""
    def __init__(self, arr: List[T], func: Callable, identity: T) -> None:
        i = 1
        while i < len(arr): i <<= 1
        self.n = i
        self.tree = [identity for _ in range(self.n << 1)]
        self.func = func
        self.identity = identity
        for i in range(len(arr)):
            self.tree[self.n + i] = arr[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = func(self.tree[i << 1], self.tree[(i << 1) | 1])

    def update(self, idx: int, val: T) -> None:
        idx += self.n
        self.tree[idx] = val
        while idx > 1:
            idx >>= 1
            self.tree[idx] = self.func(self.tree[idx << 1], self.tree[(idx << 1) | 1])

    def query(self, l: int, r: int) -> T:
        ret_left = self.identity
        ret_right = self.identity
        l += self.n
        r += self.n
        while l <= r:
            if l & 1:
                ret_left = self.func(ret_left, self.tree[l])
                l += 1
            if not r & 1:
                ret_right = self.func(self.tree[r], ret_right)
                r -= 1
            l >>= 1
            r >>= 1
        return self.func(ret_left, ret_right)


import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


def main() -> None:
    input_ = sys.stdin.readline
    for _ in range(int(input_())):
        n, m = map(int, input_().split())
        indices = list(range(m, m + n))
        st: SegmentTree = SegmentTree([0] * m + [1] * n, lambda a, b: a + b, 0)
        query = list(map(int, input_().split()))
        cur = m - 1
        ans = []
        for q in query:
            q -= 1
            ans.append(str(st.query(0, indices[q] - 1)))
            st.update(indices[q], 0)
            st.update(cur, 1)
            indices[q] = cur
            cur -= 1
        print(' '.join(ans))


if __name__ == "__main__":
    main()