import json
import matplotlib.pyplot as plt

def acquire_loss(input_dir):
    with open(input_dir, "r", encoding="utf-8") as f:
        data = json.load(f)
    losses = [item["loss"] for item in data if "loss" in item]
    epochs = [item["epoch"] for item in data if "loss" in item]
    eval_losses = [item["eval_loss"] for item in data if "eval_loss" in item]
    eval_epochs=[item["epoch"] for item in data if "eval_loss" in item]
    return losses,epochs,eval_losses,eval_epochs

def plot_and_save(losses,epochs,eval_losses,eval_epochs,output_dir):
    print(len(eval_epochs),len(eval_losses))
    # 绘制 loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker=".",color="red")
    plt.plot(eval_epochs,eval_losses,marker="o",color="blue")
    plt.plot()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training_EVAL Loss Curve")
    plt.grid(True)

    # 保存
    plt.savefig(output_dir,dpi=200)
    plt.show()

'''
#多组loss拼接
input_dir = "train_results/splitline_predict/gemma3_1b_finetuned_house_generate2"
loss_list,epoch_list,eval_loss_list,eval_epoch_list=[],[],[],[]
for i in range(26): #获取多组训练损失，并拼接在一起
    if i%2==0:
        cur_input_dir=f"{input_dir}/embedding_1/emb_loss_history_{int(i/2)+1}.json"
    else:
        cur_input_dir=f"{input_dir}/lora_1/lora_loss_history_{int(i/2)+1}.json"

    losses,epochs,eval_losses,eval_epochs=acquire_loss(cur_input_dir)
    #对epoch添加步数
    epochs = [x + i for x in epochs]
    eval_epochs=[x + i for x in eval_epochs]
    #合并数据
    loss_list+=losses
    epoch_list+=epochs
    eval_loss_list+=eval_losses
    eval_epoch_list+=eval_epochs
    #break
'''

#单组loss
input_dir="five_plus_two_optimization/train_model_new/train_base_model/TRAIN_RESULTS/ernie_base_model_10/loss_history.json"
losses,epochs,eval_losses,eval_epochs=acquire_loss(input_dir)

#可视化并保存
output_dir ="five_plus_two_optimization/train_model_new/train_base_model/TRAIN_RESULTS/ernie_base_model_10/loss_curve"
plot_and_save(losses,epochs,eval_losses,eval_epochs,output_dir)
#plot_and_save(loss_list,epoch_list,eval_loss_list,eval_epoch_list,output_dir)