def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


for n in range(7):
    print(factorial(n))

for n in range(1, 7):
    print(factorial(n))