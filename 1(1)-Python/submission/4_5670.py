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