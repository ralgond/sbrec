from utils import data_partition, WarpSampler

file_path = "../data/test_user_item_1.txt"

def __unit_test_1():
    user_train, user_valid, user_test, usernum, itemnum = data_partition(file_path=file_path)

    print (user_train)
    print (user_valid)
    print (user_test)

def __unit_test_2():
    user_train, user_valid, user_test, usernum, itemnum = data_partition(file_path=file_path)
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=2, maxlen=5, n_workers=1)
    u, seq, pos, neg = sampler.next_batch()
    print (u, seq, pos, neg)

if __name__ == "__main__":
    __unit_test_2()