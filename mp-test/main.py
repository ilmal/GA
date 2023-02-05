import concurrent.futures

def main(input):
    while True:
        with open("./tmp.txt", "r") as file:
            text = file.read()
            file.close()


def init():

    with concurrent.futures.ProcessPoolExecutor() as executor:
        arr = [None] * 10000
        executor.map(main, arr)


if __name__ == "__main__":
    init()