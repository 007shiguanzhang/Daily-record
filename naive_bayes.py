import codecs
import random
import re


class NaiveBayes(object):
    def __init__(self, path="train.txt"):
        self.data = {"ham": [], "spam": []}
        self.test = {"ham": [], "spam": []}
        self.vocabulary, self.ham_pro, self.spam_pro = [], [], []
        with codecs.open(path, 'r', 'utf-8') as f:
            data = f.readlines()
            for temp in data:
                text = self.pre_process(temp)
                tag, inputs = text[0], text[1:]
                self.data[tag].append(inputs)
                # break
        self.data["ham"], self.test["ham"] = self.split(self.data["ham"], 0.2)
        self.data["spam"], self.test["spam"] = self.split(self.data["spam"], 0.2)
        self.num_ham, self.num_spam = len(self.data["ham"]), len(self.data["spam"])
        total = self.num_ham + self.num_spam
        self.pro_ham, self.pro_spam = float(self.num_ham/total), float(self.num_spam/total)
        self.pro_init = 1
        try:
            self.pro_init = self.pro_ham/self.pro_spam
        except ZeroDivisionError:
            print("除数为0，数据不足")
            assert self.pro_spam == 0

    def create(self):
        vocabulary = []
        for i in self.data["ham"] + self.data["spam"]:
            vocabulary += i
            vocabulary = list(set(vocabulary))
        self.vocabulary = vocabulary

    def train(self):
        self.create()
        ham_vec = self.statistics(self.data["ham"], self.vocabulary)
        ham_total = sum(ham_vec)
        self.ham_pro = [count/ham_total for count in ham_vec]
        spam_vec = self.statistics(self.data["spam"], self.vocabulary)
        spam_total = sum(spam_vec)
        self.spam_pro = [count/spam_total for count in spam_vec]

    def accuracy(self):
        count_ham, count_spam = 0, 0
        for temp in self.data["ham"]:
            if self.judge(temp) == "ham":
                count_ham += 1
        for temp in self.data["spam"]:
            if self.judge(temp) == "spam":
                count_spam += 1
        d = {"ham_acc": count_ham/len(self.data["ham"]),
             "spam_acc": count_spam/len(self.data["spam"]),
             "total_acc": (count_spam+count_ham)/(len(self.data["ham"])+len(self.data["spam"]))}
        return d

    def test_acc(self):
        count_ham, count_spam = 0, 0
        for temp in self.test["ham"]:
            if self.judge(temp) == "ham":
                count_ham += 1
        for temp in self.test["spam"]:
            if self.judge(temp) == "spam":
                count_spam += 1
        d = {"ham_acc": count_ham / len(self.test["ham"]),
             "spam_acc": count_spam / len(self.test["spam"]),
             "total_acc": (count_spam + count_ham) / (len(self.test["ham"]) + len(self.test["spam"]))}
        return d

    def judge(self, words):
        result = self.pro_init
        for word in words:
            if word in self.vocabulary:
                index = self.vocabulary.index(word)
                result = result * (self.ham_pro[index]/self.spam_pro[index])
            else:
                result = result * (len(self.ham_pro)/len(self.spam_pro))
        if result > 1:
            return "ham"
        else:
            return "spam"

    @staticmethod
    def statistics(words, vocabulary):
        vec = [1] * len(vocabulary)
        for temp in words:
            for word in temp:
                if word in vocabulary:
                    vec[vocabulary.index(word)] += 1
                else:
                    continue
        return vec

    @staticmethod
    def pre_process(text):
        reg = re.compile(r'[^a-zA-Z]|\d')
        words = reg.split(text)
        result = [word.lower() for word in words if len(word) > 0]
        return result

    @staticmethod
    def split(data, rate):
        random_index = random.sample(range(len(data)), int(len(data)*rate))
        data_train, data_test = [], []
        for index, temp in enumerate(data):
            if index in random_index:
                data_test.append(temp)
            else:
                data_train.append(temp)
        return data_train, data_test

if __name__ == "__main__":
    test = NaiveBayes()
    test.train()
    result = test.accuracy()
    print("train_acc:", result)
    dec_acc = test.test_acc()
    print("dec_acc:", dec_acc)
    test_txt = random.choice(test.data["ham"])
    print("test文本输入：{0}, 判断结果:{1}".format(test_txt, test.judge(test_txt)))
