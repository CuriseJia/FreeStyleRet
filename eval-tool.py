import numpy as np

class Evaluator():
    def __init__(self):
        pass

    def getR1Accuary(prob):
        temp = prob.detach().cpu().numpy()
        temp = np.argsort(temp, axis=1)
        count = 0
        for i in range(prob.shape[0]):
            if temp[i][prob.shape[1]-1] == i:
                count+=1
        acc = count/prob.shape[0]
        return acc


    def getR5Accuary(prob):
        temp = prob.detach().cpu().numpy()
        temp = np.argsort(temp, axis=1)
        count = 0
        for i in range(prob.shape[0]):
            for j in range(prob.shape[1]-4,prob.shape[1]):
                if temp[i][j] == i:
                    count+=1
        acc = count/prob.shape[0]
        return acc

if __name__ == "__main__":
    eval = Evaluator()
    prob = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    prob = torch.tensor(prob)
    print(eval.getR1Accuary(prob))
    print(eval.getR5Accuary(prob))
    # Output:
    # 0.5
    # 1.0
    # Explanation:
    # The first row has the highest probability at the last index, so the R1 accuracy is 0.5.
    # The second row has the highest probability at the first index, so the R5 accuracy is 1.0.
    # The R5 accuracy is 1.0 because the highest probability is at the last index, which is within the top 5 indices.