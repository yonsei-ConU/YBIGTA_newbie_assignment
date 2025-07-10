from lib import Trie
import sys


"""
TODO:
- 일단 Trie부터 구현하기
- count 구현하기
- main 구현하기
"""


def count(trie: Trie, query_seq: str) -> int:
    """
    trie - 이름 그대로 trie
    query_seq - 단어 ("hello", "goodbye", "structures" 등)

    returns: query_seq의 단어를 입력하기 위해 버튼을 눌러야 하는 횟수
    """
    pass


def main() -> None:
    input_ = sys.stdin.readline
    while True:
        try:
            N = int(input_())
        except:
            break

        trie = Trie()
        words = [input_() for _ in range(N)]
        for w in words:
            trie.push(w)
        print(f"{trie.dfs(0) / N:.2f}")


if __name__ == "__main__":
    main()