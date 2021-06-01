import random
import copy


if __name__ == '__main__':
    print("Test cases for 'Trace'")
    alphabet = ['a', 'b', 'c']
    t = Trace()
    k = 5
    for _ in range(k):
        a = random.choice(alphabet)
        t = t.push(a)

    history = copy.deepcopy(t.history)
    print(t.history)

    for i in range(1, k+1):
        a, t = t.pop()
        assert a == history[-i], f"a != history[-{i}] ({a} != {history[-i]})"
    assert len(t.history) == 0