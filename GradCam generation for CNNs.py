def get_grad_cam(image, model, layer_name, channel_size):

    from keras import backend as K

    image_1 = np.expand_dims(image, axis=0)
    predict = model.predict(image_1)
    target_class = np.argmax(predict[0])

    last_conv = model.get_layer(layer_name)
    grads = K.gradients(model.output[:,target_class],last_conv.output)[0]

    pooled_grads = K.mean(grads,axis=(0,1,2))
    iterate = K.function([model.input],[pooled_grads,last_conv.output[0]])
    pooled_grads_value,conv_layer_output = iterate([image_1])

    for i in range(channel_size):
        conv_layer_output[:,:,i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output,axis=-1)

    #ReLu
    for x in range(heatmap.shape[0]):
        for y in range(heatmap.shape[1]):
            heatmap[x,y] = np.max(heatmap[x,y],0)

    #Normalize
    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)
    plt.imshow(heatmap)
    plt.show()

    #Overlay
    upsample = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    plt.imshow(np.interp(image_1[0], [np.min(image_1[0]), np.max(image_1[0])], [0, 1]))
    plt.show()
    plt.imshow(np.interp(image_1[0], [np.min(image_1[0]), np.max(image_1[0])], [0, 1]))
    plt.imshow(upsample,alpha=0.45)
    plt.show()
