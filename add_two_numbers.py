def add_two_numbers(num1, num2):
    return num1 + num2

if __name__ == '__main__':
    a = float(input('Enter the first number: '))
    b = float(input('Enter the second number: '))
    result = add_two_numbers(a, b)
    print('The sum of the two numbers is:', result)
