from lib import Trie
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