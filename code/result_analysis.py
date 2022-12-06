import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay


def plot_result(model_name, real_y, predict_y, labels, log_dir=None, plot=True):
    report_dir = classification_report(y_true=real_y, y_pred=predict_y, labels=labels, digits=4, output_dict=True)
    result = "=== {} report ===\n".format(model_name)
    result += "=== Precision {:.2f}%   Recall {:.2f}%    Sensitivity {:.2f}%    Specificity {:.2f}%    F1 {:.4f}\n".format(
            report_dir["1"]["precision"],
            report_dir["1"]["recall"],
            report_dir["2"]["precision"],
            report_dir["2"]["recall"],
            report_dir["1"]["f1-score"]
        )
    if log_dir != None:
        log = open(log_dir + "/log.txt", "a+")
        log.write(result+"\n")
        log.close()
    if plot:
        print(result)



def plot_confusion_matrix(real_y, predict_y, labels):
    for cm in multilabel_confusion_matrix(real_y, predict_y, labels=labels):
        print(cm)
        ConfusionMatrixDisplay(cm).plot()
        plt.show()

def tight_layout(model, predict_x, predict_y,labels, save_dir=None, plot=False):
    ConfusionMatrixDisplay.from_estimator(
        model, predict_x, predict_y, display_labels=labels, xticks_rotation="vertical"
    )
    plt.tight_layout()
    if save_dir != None:
        plt.savefig(save_dir)
    if plot:
        plt.show()
    plt.close("all")
