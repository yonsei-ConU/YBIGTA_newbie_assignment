from lib import SegmentTree
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