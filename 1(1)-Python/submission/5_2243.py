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
    def __init__(self, arr: List[T], func: U, identity: T) -> None:
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
    n = int(input_())
    st = SegmentTree([0] * 10 ** 6, lambda a, b: a + b, 0)

    for _ in range(n):
        l = list(map(int, input_().split()))
        if l[0] == 1:
            lo = -1
            hi = 10 ** 6
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if st.query(0, mid) >= l[1]:
                    hi = mid
                else:
                    lo = mid
            res = hi
            print(hi + 1)
            st.update(hi, st.tree[res + 2 ** 20] - 1)
        else:
            flavor, delta = l[1:]
            flavor -= 1
            current_amount = st.tree[flavor + 2 ** 20]
            st.update(flavor, current_amount + delta)


if __name__ == "__main__":
    main()