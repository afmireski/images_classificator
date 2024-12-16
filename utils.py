# Importa libs úteis para avaliação dos modelos
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def show_predict_infos(y, predict, title="", cmap="Blues"):
    accuracy = accuracy_score(y, predict)
    percent = accuracy * 100
    print(f"A acurácia no conjunto de testes: {percent:.2f}%")

    # Mostra um relatório com as métricas de classificação por classe e as métricas calculadas sobre o conjunto todo.
    print(classification_report(y, predict))

    ConfusionMatrixDisplay.from_predictions(y, predict, colorbar=False, cmap=cmap)
    if len(title) > 0:
        plt.title(f"Matriz de Confusão {title}")
    plt.xlabel("Rótulo Previsto")
    plt.ylabel("Rótulo Real")
    plt.show()
    
