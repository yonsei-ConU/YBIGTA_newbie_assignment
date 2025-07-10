from lib import SegmentTree
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
        st = SegmentTree([0] * m + [1] * n, lambda a, b: a + b, 0)
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