def main():
    file = open("test/data.spec")
    spec = file.read()
    print(spec)
    file.close()

    file = open("test/data.dat")
    data = file.read()
    print(data)
    file.close()


class Attribute:
    attrVals = None

    def __init__(self, isContinuous, name, values):
        self.isContinuous = isContinuous
        self.name = name
        if self.isContinuous == False :
            self.attrVals = values

if __name__ == "__main__" :
    main()