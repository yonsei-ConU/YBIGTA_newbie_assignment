from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Iterable


"""
TODO:
- Trie.push 구현하기
- (필요할 경우) Trie에 추가 method 구현하기
"""


T = TypeVar("T")


@dataclass
class TrieNode(Generic[T]):
    body: Optional[T] = None
    children: list[int] = field(default_factory=lambda: [])
    is_end: bool = False


class Trie(list[TrieNode[T]]):
    def __init__(self) -> None:
        super().__init__()
        self.append(TrieNode(body=None))

    def push(self, seq: Iterable[T]) -> None:
        """
        seq: T의 열 (list[int]일 수도 있고 str일 수도 있고 등등...)
        action: trie에 seq을 저장하기
        """
        cur_idx = 0
        for element in seq:
            found = False
            for child_idx in self[cur_idx].children:
                if self[child_idx].body == element:
                    cur_idx = child_idx
                    found = True
                    break
            if not found:
                new_node = TrieNode(body=element)
                self.append(new_node)
                new_idx = len(self) - 1
                self[cur_idx].children.append(new_idx)
                cur_idx = new_idx
        self[cur_idx].is_end = True

    def dfs(self, root: int) -> int:
        """
        implemented for problem 5670
        :param root: the index of root node

        Need not use `count` function, instead you can just run this once and get the correct answer
        """
        ret = 0

        def doDFS(cur: int, depth: int, auto: bool = False) -> None:
            """
            :param cur: the index of the current node
            :param depth: the depth (distance between the root and cur)
            :param auto: to check this node is the only child
            """
            nonlocal ret
            node = self[cur]
            if node.body == '\n':
                if not auto:
                    depth -= 1
                ret += depth
            elif len(node.children) == 1 and cur != 0:
                child_idx = node.children[0]
                doDFS(child_idx, depth, True)
            else:
                for child_idx in node.children:
                    doDFS(child_idx, depth + 1)

        doDFS(root, 0)
        return ret



import sys


"""
TODO:
- 일단 lib.py의 Trie Class부터 구현하기
- main 구현하기

힌트: 한 글자짜리 자료에도 그냥 str을 쓰기에는 메모리가 아깝다...
"""
ans = 1


def dfs(start, end, depth, names, fact, MOD):
    """
    Recursive binary search solution
    I guess this solution is much easier than solutions involving trie,
    since this solution requires much less memory than the trie solution.
    Implementing this solution is also not very hard.
    Because this is one solution that YBIGTA graders might not be expecting,
    I will provide a detailed description of how this solution works.

    :param start: start of the interval I'm currently looking at.
    :param end: end of the interval I'm currently looking at.
    the interval is represented by [start, end].
    :param depth: how deep is the current branch. Relevant to the index of the string.
    :param names: given input.
    :param fact: precomputed array of factorials modulo MOD.
    :param MOD: 10 ** 9 + 7

    THe solution assumes that the list `names` is already sorted.
    I achieve that assumption by sorting `names` beforehand.
    Let `names` be sorted.
    During a recursive call, all names in the interval [start, end]
    is identical in the first `depth` characters.
    So in a recursive call it is the turn to compare the `depth`-th character of each name.
    The rest is pretty much the same as the Trie solution.
    We identify, for each character (including empty string),
    how many strings in the interval has that character as the `depth`-th character.
    This is done by binary search, so it runs in O(26log(end-start)) time.
    If there is a string in the interval whose `depth`-th character is A,
    then all strings in the interval whose `depth`-th character is A should be in consecutive order.
    But for strings in the interval whose `depth`-th character does not exist, they can be anywhere in the interval.
    So I conclude that the number of ways is equal to the factorial of
    (the number of strings in the interval whose `depth`-th character does not exist) +
    (the number of alphabets that there exists a string in the interval whose `depth`-th character is that alphabet).
    Using some combinatorics idea, I conclude that the real answer is
    the product of all answers in every recursive call.
    """
    global ans
    if start == end: return
    lo = start - 1
    hi = end + 1
    while lo + 1 < hi:
        mid = (lo + hi) >> 1
        if len(names[mid]) == depth:
            lo = mid
        else:
            hi = mid
    left = hi
    mult = hi - start
    for c in range(65, 65 + 26):
        d = chr(c)
        lo = left - 1
        hi = end + 1
        while lo + 1 < hi:
            mid = (lo + hi) >> 1
            if names[mid][depth] == d:
                lo = mid
            else:
                hi = mid
        if lo > left - 1:
            dfs(left, lo, depth + 1, names, fact, MOD)
            mult += 1
        left = hi
    ans = ans * fact[mult] % MOD


def main() -> None:
    N = int(sys.stdin.readline())
    MOD = 10 ** 9 + 7
    fact = [1, 1]
    for i in range(2, 3001):
        fact.append(fact[i - 1] * i % MOD)
    names = sorted([sys.stdin.readline().rstrip() for _ in range(N)])
    dfs(0, N - 1, 0, names, fact, MOD)
    print(ans)


if __name__ == "__main__":
    main()