import matplotlib.pyplot as plt


val_acc = [0.08888889104127884, 0.09444444626569748, 0.10000000149011612, 0.12222222238779068, 0.16111111640930176, 0.14444445073604584, 0.15555556118488312, 0.15555556118488312, 0.1944444477558136, 0.1666666716337204, 0.22777777910232544, 0.2222222238779068, 0.20000000298023224, 0.22777777910232544, 0.2222222238779068, 0.21666666865348816, 0.23888888955116272, 0.21666666865348816, 0.20555555820465088, 0.20000000298023224]
val_acc2 = [0.12777778506278992, 0.17222222685813904, 0.17777778208255768, 0.15000000596046448, 0.18888889253139496, 0.21111111342906952, 0.20555555820465088, 0.18333333730697632, 0.17777778208255768, 0.1666666716337204, 0.20000000298023224, 0.1666666716337204, 0.15555556118488312, 0.15000000596046448, 0.1388888955116272, 0.12777778506278992, 0.17777778208255768, 0.1388888955116272, 0.15555556118488312, 0.15000000596046448]
val_acc3 = 0.05000000074505806, 0.07222222536802292, 0.20000000298023224, 0.09444444626569748, 0.18888889253139496, 0.16111111640930176, 0.15000000596046448, 0.18333333730697632, 0.18888889253139496, 0.17222222685813904, 0.20000000298023224, 0.18333333730697632, 0.17777778208255768, 0.17777778208255768, 0.20000000298023224, 0.14444445073604584, 0.21111111342906952, 0.18333333730697632, 0.17222222685813904, 0.1944444477558136
val_acc_bert = [0.29444444, 0.3, 0.38333333, 0.3611111, 0.3888889, 0.41666666, 0.43333334, 0.41666666]
epoch = list(range(8))

print(val_acc)
print(epoch)

plt.plot(epoch, val_acc_bert)
plt.ylabel('Validation Accuracy')
plt.xlabel('Training Epoch')
plt.tight_layout()
plt.savefig('../images/metrics/val_acc_bert.png')

