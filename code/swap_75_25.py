import pandas as pd

# Dictionary of result files
files = {
    #!!! When swaping MUCAC Be careful only swap it 
    
    # "CIFAR10_AllCNN": r"C:/Temp/Unlearning/Results/Cifar 10 AllCNN/compiled_results_MIAU.csv",
    # "CIFAR10_ResNet": r"C:/Temp/Unlearning/Results/Cifar 10 Resnet/compiled_results_MIAU.csv",
    # "CIFAR20_AllCNN": r"C:/Temp/Unlearning/Results/Cifar 20 AllCNN/compiled_results_MIAU.csv",
    # "CIFAR20_ResNet": r"C:/Temp/Unlearning/Results/Cifar 20 Resnet/compiled_results_MIAU.csv",
    # "CIFAR10_ViT": r"C:/Temp/Unlearning/Results/Cifar 10 ViT/compiled_results_MIAU.csv",
    # "MNIST_ResNet": r"C:/Temp/Unlearning/Results/MNIST Resnet/compiled_results_MIAU.csv",
    # "MNIST_AllCNN": r"C:/Temp/Unlearning/Results/MNIST AllCNN/compiled_results_MIAU.csv",
    # "MUCAC_ResNet": r"C:/Temp/Unlearning/Results/MUCAC Resnet/compiled_results_MIAU.csv"
    
    
    # "CIFAR10_ResNet_Underfitted": r"C:/Temp/Unlearning/Results/Underfitted/compiled_results_MIAU.csv",
    # "CIFAR10_ResNet_Overfitted": r"C:/Temp/Unlearning/Results/Overfitted/compiled_results_MIAU.csv"
    
    #"CIFAR10_ResNet_Saliency": r"C:\Temp\Unlearning\Data Appendix\Cifar 10 Resnet Saliency\compiled_results_MIAU.csv",
    #"CIFAR20_AllCNN_SubClass": r"C:\Temp\Unlearning\Data Appendix\Cifar 20 AllCNN SubClass\compiled_results_MIAU.csv",
    #"CIFAR20_AllCNN_FullClass": r"C:\Temp\Unlearning\Data Appendix\Cifar 20 AllCNN FullClass\compiled_results_MIAU.csv"





#    "CIFAR10_AllCNN": r"C:/Temp/Unlearning/Data Appendix/Cifar 10 AllCNN/compiled_results_MIAU_0.5.csv",
#     "CIFAR10_ResNet": r"C:/Temp/Unlearning/Data Appendix/Cifar 10 Resnet/compiled_results_MIAU_0.5.csv",
#      "CIFAR10_ResNet_Saliency": r"C:\Temp\Unlearning\Data Appendix\Cifar 10 Resnet Saliency\compiled_results_MIAU_0.5.csv",

#     "CIFAR20_AllCNN": r"C:/Temp/Unlearning/Data Appendix/Cifar 20 AllCNN/compiled_results_MIAU_0.5.csv",
#       "CIFAR20_AllCNN_SubClass": r"C:\Temp\Unlearning\Data Appendix\Cifar 20 AllCNN SubClass\compiled_results_MIAU_0.5.csv",
#     "CIFAR20_AllCNN_FullClass": r"C:\Temp\Unlearning\Data Appendix\Cifar 20 AllCNN FullClass\compiled_results_MIAU_0.5.csv",
    
#     "CIFAR20_ResNet": r"C:/Temp/Unlearning/Data Appendix/Cifar 20 Resnet/compiled_results_MIAU_0.5.csv",
#     "CIFAR10_ViT": r"C:/Temp/Unlearning/Data Appendix/Cifar 10 ViT/compiled_results_MIAU_0.5.csv",
#     "MNIST_ResNet": r"C:/Temp/Unlearning/Data Appendix/MNIST Resnet/compiled_results_MIAU_0.5.csv",
#     "MNIST_AllCNN": r"C:/Temp/Unlearning/Data Appendix/MNIST AllCNN/compiled_results_MIAU_0.5.csv",
#     "MUCAC_ResNet": r"C:/Temp/Unlearning/Data Appendix/MUCAC Resnet/compiled_results_MIAU_0.5.csv",
    
    "CIFAR10_ResNet_Underfitted": r"C:/Temp/Unlearning/Data Appendix/Underfitted/compiled_results_MIAU_0.5.csv",
    "CIFAR10_ResNet_Overfitted": r"C:/Temp/Unlearning/Data Appendix/Overfitted/compiled_results_MIAU_0.5.csv"
}

for name, path in files.items():
    try:
        df = pd.read_csv(path)

        # Swap labels
        df['unlearning'] = df['unlearning'].replace({
            'retrain25': 'TEMP_SWAP_TAG',
            'retrain75': 'retrain25'
        })
        df['unlearning'] = df['unlearning'].replace({
            'TEMP_SWAP_TAG': 'retrain75'
        })

        # Overwrite file (optional: save to new file)
        df.to_csv(path, index=False)
        print(f"Swapped retrain25 <-> retrain75 in {name}")
        
    except Exception as e:
        print(f"Error processing {name}: {e}")
