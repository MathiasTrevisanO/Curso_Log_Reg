
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# le o dataset 
dataset = pd.read_csv('breast_cancer.csv')
print(dataset)

# seleciona as entradas e saidas desejadas
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

def main():
    # separar o dataset em dados de treino e dados de teste
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state= 0)
    # aplica a funcao de regressao logistica para treinar o modelo   
    classifier = LogisticRegression()   
    classifier.fit(X_train, y_train)
    
    # deploy do modelo realizando a predicao se é benigno ou maligno pelos dados de teste
    y_pred = classifier.predict(X_test)
    
    # chama o metodo de confusion_matrix para visualizar a matriz onde os casos deram 
    # benignos e malignos, e os casos onde deram falsos positivos e falsos negativos
    # (para benigno e maligno).
    cm = confusion_matrix(y_test,y_pred)
    print(cm)
    # calculo da acurácia
    print("Acurácia: ",round((84+47)/(84+47+3+3),2))
    
    # k-fold Cross validation
    accuracies = cross_val_score(estimator= classifier, X = X_train, y = y_train, cv= 10)
    print("----Cross Validation----")
    print("Acurácia: {:.2f}%".format(accuracies.mean()*100))
    print("Desvio padrão: {:.2f}%".format(accuracies.std()*100))
    
if __name__ == "__main__":
    main()