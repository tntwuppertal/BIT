from keras.models import Model
from keras.layers import Input, Add, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Dense, Flatten, Concatenate
from keras.optimizers import Adam
from keras import regularizers


################################################################################################
# Referenzmodel
# img_rows: Inputimage height (int)
# img_cols: Inputimage width (int)
# nbInputChannel: Number of Inputchannels (1: monochrome, 3: RGB) (int)
# nbClasses: Number of Classes (int)
# fak: Channelfactor (modelparameter) (int)
# batch: use of batchnormalization (bool)
# drop: use of dropout (bool)
# lr: learningrate (float)
def ref_model(img_rows,img_cols,nbInputChannel,nbClasses,fak,batch,drop,lr):
    inp = Input(( img_rows, img_cols,nbInputChannel))
    wDecay = 0.01# weight decay
    aDecay = 0#activation decay
    droprate = 0.35
		
    # main
    conv1 = Conv2D(8*fak, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv1_1')(inp)
    if batch:
        conv1=BatchNormalization(name='conv1_b1')(conv1)
    conv1=Activation('relu')(conv1)
    if drop:
        conv1 = Dropout(droprate)(conv1)    
    conv1 = Conv2D(8*fak, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv1_2')(conv1)
    if batch:
        conv1=BatchNormalization(name='conv1_b2')(conv1)
    conv1=Activation('relu')(conv1)
    if drop:
        conv1 = Dropout(droprate)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(16*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv2_1')(pool1)
    if batch:
        conv2=BatchNormalization(name='conv2_b1')(conv2)
    conv2=Activation('relu')(conv2)
    if drop:
        conv2 = Dropout(droprate)(conv2)    
    conv2 = Conv2D(16*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv2_2')(conv2)
    if batch:
        conv2=BatchNormalization(name='conv2_b2')(conv2)
    conv2=Activation('relu')(conv2)
    if drop:
        conv2 = Dropout(droprate)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(32*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv3_1')(pool2)
    if batch:
        conv3=BatchNormalization(name='conv3_b1')(conv3)
    conv3=Activation('relu')(conv3)
    if drop:
        conv3 = Dropout(droprate)(conv3)    
    conv3 = Conv2D(32*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv3_2')(conv3)
    if batch:
        conv3=BatchNormalization(name='conv3_b2')(conv3)
    conv3=Activation('relu')(conv3)
    if drop:
        conv3 = Dropout(droprate)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(64*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv4_1')(pool3)
    if batch:
        conv4=BatchNormalization(name='conv4_b1')(conv4)
    conv4=Activation('relu')(conv4)
    if drop:
        conv4 = Dropout(droprate)(conv4)    
    conv4 = Conv2D(64*fak, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv4_2')(conv4)
    if batch:
        conv4=BatchNormalization(name='conv4_b2')(conv4)
    conv4=Activation('relu')(conv4)
    if drop:
        conv4 = Dropout(droprate)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    flat1 = Flatten()(pool4)
    dense1 = Dense(64, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='dense1')(flat1)
    if batch:
        dense1=BatchNormalization(name='dense1_b1')(dense1)
    dense1=Activation('relu')(dense1)
    if drop:
        dense1 = Dropout(droprate)(dense1)
    dense2 = Dense(nbClasses,activation='softmax',name='dense2')(dense1)

    
    model = Model(inputs=inp, outputs=dense2)
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['acc'])
    return model

################################################################################################
# Multi-View Fusion Neural Network (MVFNN)
# img_rows: Inputimage height (int)
# img_cols: Inputimage width (int)
# nbInputChannel: Number of Inputchannels (1: monochrome, 3: RGB) (int)
# nbClasses: Number of Classes (int)
# fak: Channelfactor (modelparameter) (int)
# batch: use of batchnormalization (bool)
# drop: use of dropout (bool)
# lr: learningrate (float)
def mvfnn(img_rows,img_cols,nbInputChannel,nbClasses,fak,batch,drop,lr):
    inp_1 = Input(( img_rows, img_cols,nbInputChannel))
    inp_2 = Input(( img_rows, img_cols,nbInputChannel))
    wDecay = 0.01# weight decay
    aDecay = 0#activation decay
    droprate = 0.35

	
    # side branch layers (shared weights layers)
    conv1_1 = Conv2D(8*fak, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv1_1')
    
    conv1_2 = Conv2D(8*fak, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv1_2')
    
#################################################################
    conv2_1 = Conv2D(16*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv2_1')
    
    conv2_2 = Conv2D(16*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv2_2')
    
##################################################################
    conv3_1 = Conv2D(32*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv3_1')
    
    conv3_2 = Conv2D(32*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv3_2')

##################################################################
    conv4_1 = Conv2D(64*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv4_1')
    
    conv4_2 = Conv2D(64*fak, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv4_2')

    
    
    
    # inp_1 branch
    R1_conv1 = conv1_1(inp_1)
    if batch:
        R1_conv1=BatchNormalization(name='R1_conv1_b1')(R1_conv1)
    R1_conv1=Activation('relu')(R1_conv1)
    if drop:
        R1_conv1 = Dropout(droprate)(R1_conv1)    
    R1_conv1 = conv1_2(R1_conv1)
    if batch:
        R1_conv1=BatchNormalization(name='R1_conv1_b2')(R1_conv1)
    R1_conv1=Activation('relu')(R1_conv1)
    if drop:
        R1_conv1 = Dropout(droprate)(R1_conv1)
    R1_pool1 = MaxPooling2D(pool_size=(2, 2))(R1_conv1)
#################################################################
    R1_conv2 = conv2_1(R1_pool1)
    if batch:
        R1_conv2=BatchNormalization(name='R1_conv2_b1')(R1_conv2)
    R1_conv2=Activation('relu')(R1_conv2)
    if drop:
        R1_conv2 = Dropout(droprate)(R1_conv2)    
    R1_conv2 = conv2_2(R1_conv2)
    if batch:
        R1_conv2=BatchNormalization(name='R1_conv2_b2')(R1_conv2)
    R1_conv2=Activation('relu')(R1_conv2)
    if drop:
        R1_conv2 = Dropout(droprate)(R1_conv2)
    R1_pool2 = MaxPooling2D(pool_size=(2, 2))(R1_conv2)
##################################################################
    R1_conv3 = conv3_1(R1_pool2)
    if batch:
        R1_conv3=BatchNormalization(name='R1_conv3_b1')(R1_conv3)
    R1_conv3=Activation('relu')(R1_conv3)
    if drop:
        R1_conv3 = Dropout(droprate)(R1_conv3)    
    R1_conv3 = conv3_2(R1_conv3)
    if batch:
        R1_conv3=BatchNormalization(name='R1_conv3_b2')(R1_conv3)
    R1_conv3=Activation('relu')(R1_conv3)
    if drop:
        R1_conv3 = Dropout(droprate)(R1_conv3)
    R1_pool3 = MaxPooling2D(pool_size=(2, 2))(R1_conv3)
##################################################################
    R1_conv4 = conv4_1(R1_pool3)
    if batch:
        R1_conv4=BatchNormalization(name='R1_conv4_b1')(R1_conv4)
    R1_conv4=Activation('relu')(R1_conv4)
    if drop:
        R1_conv4 = Dropout(droprate)(R1_conv4)    
    R1_conv4 = conv4_2(R1_conv4)
    if batch:
        R1_conv4=BatchNormalization(name='R1_conv4_b2')(R1_conv4)
    R1_conv4=Activation('relu')(R1_conv4)
    if drop:
        R1_conv4 = Dropout(droprate)(R1_conv4)
    
    
    
    # inp_2 branch
    R2_conv1 = conv1_1(inp_2)
    if batch:
        R2_conv1=BatchNormalization(name='R2_conv1_b1')(R2_conv1)
    R2_conv1=Activation('relu')(R2_conv1)
    if drop:
        R2_conv1 = Dropout(droprate)(R2_conv1)    
    R2_conv1 = conv1_2(R2_conv1)
    if batch:
        R2_conv1=BatchNormalization(name='R2_conv1_b2')(R2_conv1)
    R2_conv1=Activation('relu')(R2_conv1)
    if drop:
        R2_conv1 = Dropout(droprate)(R2_conv1)
    R2_pool1 = MaxPooling2D(pool_size=(2, 2))(R2_conv1)
#################################################################
    R2_conv2 = conv2_1(R2_pool1)
    if batch:
        R2_conv2=BatchNormalization(name='R2_conv2_b1')(R2_conv2)
    R2_conv2=Activation('relu')(R2_conv2)
    if drop:
        R2_conv2 = Dropout(droprate)(R2_conv2)    
    R2_conv2 = conv2_2(R2_conv2)
    if batch:
        R2_conv2=BatchNormalization(name='R2_conv2_b2')(R2_conv2)
    R2_conv2=Activation('relu')(R2_conv2)
    if drop:
        R2_conv2 = Dropout(droprate)(R2_conv2)
    R2_pool2 = MaxPooling2D(pool_size=(2, 2))(R2_conv2)
##################################################################
    R2_conv3 = conv3_1(R2_pool2)
    if batch:
        R2_conv3=BatchNormalization(name='R2_conv3_b1')(R2_conv3)
    R2_conv3=Activation('relu')(R2_conv3)
    if drop:
        R2_conv3 = Dropout(droprate)(R2_conv3)    
    R2_conv3 = conv3_2(R2_conv3)
    if batch:
        R2_conv3=BatchNormalization(name='R2_conv3_b2')(R2_conv3)
    R2_conv3=Activation('relu')(R2_conv3)
    if drop:
        R2_conv3 = Dropout(droprate)(R2_conv3)
    R2_pool3 = MaxPooling2D(pool_size=(2, 2))(R2_conv3)
##################################################################
    R2_conv4 = conv4_1(R2_pool3)
    if batch:
        R2_conv4=BatchNormalization(name='R2_conv4_b1')(R2_conv4)
    R2_conv4=Activation('relu')(R2_conv4)
    if drop:
        R2_conv4 = Dropout(droprate)(R2_conv4)    
    R2_conv4 = conv4_2(R2_conv4)
    if batch:
        R2_conv4=BatchNormalization(name='R2_conv4_b2')(R2_conv4)
    R2_conv4=Activation('relu')(R2_conv4)
    if drop:
        R2_conv4 = Dropout(droprate)(R2_conv4)
      
    
    # main branch aka fusion branch
    M_fuse1 = Add()([R1_conv1, R2_conv1])
    M_conv1 = Conv2D(2*8*fak, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='M_conv1_1')(M_fuse1)
    if batch:
        M_conv1=BatchNormalization(name='M_conv1_b1')(M_conv1)
    M_conv1=Activation('relu')(M_conv1)
    if drop:
        M_conv1 = Dropout(droprate)(M_conv1)    

    M_pool1 = MaxPooling2D(pool_size=(2, 2))(M_conv1)
    ##########################################################
    M_fuse2 = Add()([M_pool1, R1_conv2, R2_conv2])
    M_conv2 = Conv2D(2*16*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='M_conv2_1')(M_fuse2)
    if batch:
        M_conv2=BatchNormalization(name='M_conv2_b1')(M_conv2)
    M_conv2=Activation('relu')(M_conv2)
    if drop:
        M_conv2 = Dropout(droprate)(M_conv2)    

    M_pool2 = MaxPooling2D(pool_size=(2, 2))(M_conv2)
   #############################################################
    M_fuse3 = Add()([M_pool2, R1_conv3, R2_conv3])
    M_conv3 = Conv2D(2*32*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='M_conv3_1')(M_fuse3)
    if batch:
        M_conv3=BatchNormalization(name='M_conv3_b1')(M_conv3)
    M_conv3=Activation('relu')(M_conv3)
    if drop:
        M_conv3 = Dropout(droprate)(M_conv3)    

    M_pool3 = MaxPooling2D(pool_size=(2, 2))(M_conv3)   
    #############################################################
    M_fuse4 = Add()([M_pool3, R1_conv4, R2_conv4])
    M_conv4 = Conv2D(2*64*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='M_conv4_1')(M_fuse4)
    if batch:
        M_conv4=BatchNormalization(name='M_conv4_b1')(M_conv4)
    M_conv4=Activation('relu')(M_conv4)
    if drop:
        M_conv4 = Dropout(droprate)(M_conv4)    

    M_pool4 = MaxPooling2D(pool_size=(2, 2))(M_conv4)    
    
    
    
    M_flat1 = Flatten()(M_pool4)
    M_dense1 = Dense(64, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='M_dense1')(M_flat1)
    if batch:
        M_dense1=BatchNormalization(name='M_dense1_b1')(M_dense1)
    M_dense1=Activation('relu')(M_dense1)
    if drop:
        M_dense1 = Dropout(droprate)(M_dense1)
    M_dense2 = Dense(nbClasses,activation='softmax',name='M_dense2')(M_dense1)

    
    model = Model(inputs=[inp_1, inp_2], outputs=M_dense2)
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['acc'])
    return model


################################################################################################
# early fusion (concatenation of images)
# img_rows: Inputimage height (int)
# img_cols: Inputimage width (int)
# nbInputChannel: Number of Inputchannels (1: monochrome, 3: RGB) (int)
# nbClasses: Number of Classes (int)
# fak: Channelfactor (modelparameter) (int)
# batch: use of batchnormalization (bool)
# drop: use of dropout (bool)
# lr: learningrate (float)
def concat_fusion(img_rows,img_cols,nbInputChannel,nbClasses,fak,batch,drop,lr):
    inp_1 = Input(( img_rows, img_cols,nbInputChannel))
    inp_2 = Input(( img_rows, img_cols,nbInputChannel))
    wDecay = 0.01# weight decay
    aDecay = 0#activation decay
    droprate = 0.35


    concatFused = Concatenate()([inp_1, inp_2])	
	# main
    conv1 = Conv2D(8*fak, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv1_1')(concatFused)
    if batch:
        conv1=BatchNormalization(name='conv1_b1')(conv1)
    conv1=Activation('relu')(conv1)
    if drop:
        conv1 = Dropout(droprate)(conv1)    
    conv1 = Conv2D(8*fak, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv1_2')(conv1)
    if batch:
        conv1=BatchNormalization(name='conv1_b2')(conv1)
    conv1=Activation('relu')(conv1)
    if drop:
        conv1 = Dropout(droprate)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(16*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv2_1')(pool1)
    if batch:
        conv2=BatchNormalization(name='conv2_b1')(conv2)
    conv2=Activation('relu')(conv2)
    if drop:
        conv2 = Dropout(droprate)(conv2)    
    conv2 = Conv2D(16*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv2_2')(conv2)
    if batch:
        conv2=BatchNormalization(name='conv2_b2')(conv2)
    conv2=Activation('relu')(conv2)
    if drop:
        conv2 = Dropout(droprate)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(32*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv3_1')(pool2)
    if batch:
        conv3=BatchNormalization(name='conv3_b1')(conv3)
    conv3=Activation('relu')(conv3)
    if drop:
        conv3 = Dropout(droprate)(conv3)    
    conv3 = Conv2D(32*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv3_2')(conv3)
    if batch:
        conv3=BatchNormalization(name='conv3_b2')(conv3)
    conv3=Activation('relu')(conv3)
    if drop:
        conv3 = Dropout(droprate)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(64*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv4_1')(pool3)
    if batch:
        conv4=BatchNormalization(name='conv4_b1')(conv4)
    conv4=Activation('relu')(conv4)
    if drop:
        conv4 = Dropout(droprate)(conv4)    
    conv4 = Conv2D(64*fak, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv4_2')(conv4)
    if batch:
        conv4=BatchNormalization(name='conv4_b2')(conv4)
    conv4=Activation('relu')(conv4)
    if drop:
        conv4 = Dropout(droprate)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    flat1 = Flatten()(pool4)
    dense1 = Dense(64, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='dense1')(flat1)
    if batch:
        dense1=BatchNormalization(name='dense1_b1')(dense1)
    dense1=Activation('relu')(dense1)
    if drop:
        dense1 = Dropout(droprate)(dense1)
    dense2 = Dense(nbClasses,activation='softmax',name='dense2')(dense1)

    
    model = Model(inputs=[inp_1, inp_2], outputs=dense2)    
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['acc'])
    return model


################################################################################################
# late fusion (FC fusionhead)
# img_rows: Inputimage height (int)
# img_cols: Inputimage width (int)
# nbInputChannel: Number of Inputchannels (1: monochrome, 3: RGB) (int)
# nbClasses: Number of Classes (int)
# fak: Channelfactor (modelparameter) (int)
# batch: use of batchnormalization (bool)
# drop: use of dropout (bool)
# lr: learningrate (float)
def fc_fusion(img_rows,img_cols,nbInputChannel,nbClasses,fak,batch,drop,lr):
    inp_1 = Input(( img_rows, img_cols,nbInputChannel))
    inp_2 = Input(( img_rows, img_cols,nbInputChannel))
    wDecay = 0.01# weight decay
    aDecay = 0#activation decay
    droprate = 0.35

	
    # side branch layers (shared weights layers)
    conv1_1 = Conv2D(8*fak, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv1_1')
    
    conv1_2 = Conv2D(8*fak, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv1_2')
    
#################################################################
    conv2_1 = Conv2D(16*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv2_1')
    
    conv2_2 = Conv2D(16*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv2_2')
    
##################################################################
    conv3_1 = Conv2D(32*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv3_1')
      
    conv3_2 = Conv2D(32*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv3_2')

##################################################################
    conv4_1 = Conv2D(64*fak, ((3, 3)), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv4_1')
    
    conv4_2 = Conv2D(64*fak, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='conv4_2')
    
       
    
    
    # inp_1 branch
    R1_conv1 = conv1_1(inp_1)
    if batch:
        R1_conv1=BatchNormalization(name='R1_conv1_b1')(R1_conv1)
    R1_conv1=Activation('relu')(R1_conv1)
    if drop:
        R1_conv1 = Dropout(droprate)(R1_conv1)    
    R1_conv1 = conv1_2(R1_conv1)
    if batch:
        R1_conv1=BatchNormalization(name='R1_conv1_b2')(R1_conv1)
    R1_conv1=Activation('relu')(R1_conv1)
    if drop:
        R1_conv1 = Dropout(droprate)(R1_conv1)
    R1_pool1 = MaxPooling2D(pool_size=(2, 2))(R1_conv1)
#################################################################
    R1_conv2 = conv2_1(R1_pool1)
    if batch:
        R1_conv2=BatchNormalization(name='R1_conv2_b1')(R1_conv2)
    R1_conv2=Activation('relu')(R1_conv2)
    if drop:
        R1_conv2 = Dropout(droprate)(R1_conv2)    
    R1_conv2 = conv2_2(R1_conv2)
    if batch:
        R1_conv2=BatchNormalization(name='R1_conv2_b2')(R1_conv2)
    R1_conv2=Activation('relu')(R1_conv2)
    if drop:
        R1_conv2 = Dropout(droprate)(R1_conv2)
    R1_pool2 = MaxPooling2D(pool_size=(2, 2))(R1_conv2)
##################################################################
    R1_conv3 = conv3_1(R1_pool2)
    if batch:
        R1_conv3=BatchNormalization(name='R1_conv3_b1')(R1_conv3)
    R1_conv3=Activation('relu')(R1_conv3)
    if drop:
        R1_conv3 = Dropout(droprate)(R1_conv3)    
    R1_conv3 = conv3_2(R1_conv3)
    if batch:
        R1_conv3=BatchNormalization(name='R1_conv3_b2')(R1_conv3)
    R1_conv3=Activation('relu')(R1_conv3)
    if drop:
        R1_conv3 = Dropout(droprate)(R1_conv3)
    R1_pool3 = MaxPooling2D(pool_size=(2, 2))(R1_conv3)
##################################################################
    R1_conv4 = conv4_1(R1_pool3)
    if batch:
        R1_conv4=BatchNormalization(name='R1_conv4_b1')(R1_conv4)
    R1_conv4=Activation('relu')(R1_conv4)
    if drop:
        R1_conv4 = Dropout(droprate)(R1_conv4)    
    R1_conv4 = conv4_2(R1_conv4)
    if batch:
        R1_conv4=BatchNormalization(name='R1_conv4_b2')(R1_conv4)
    R1_conv4=Activation('relu')(R1_conv4)
    if drop:
        R1_conv4 = Dropout(droprate)(R1_conv4)
    R1_conv4 = MaxPooling2D(pool_size=(2, 2))(R1_conv4)    
    R1_flat1 = Flatten()(R1_conv4)    

    
    
    
    
    # inp_2 branch
    R2_conv1 = conv1_1(inp_2)
    if batch:
        R2_conv1=BatchNormalization(name='R2_conv1_b1')(R2_conv1)
    R2_conv1=Activation('relu')(R2_conv1)
    if drop:
        R2_conv1 = Dropout(droprate)(R2_conv1)    
    R2_conv1 = conv1_2(R2_conv1)
    if batch:
        R2_conv1=BatchNormalization(name='R2_conv1_b2')(R2_conv1)
    R2_conv1=Activation('relu')(R2_conv1)
    if drop:
        R2_conv1 = Dropout(droprate)(R2_conv1)
    R2_pool1 = MaxPooling2D(pool_size=(2, 2))(R2_conv1)
#################################################################
    R2_conv2 = conv2_1(R2_pool1)
    if batch:
        R2_conv2=BatchNormalization(name='R2_conv2_b1')(R2_conv2)
    R2_conv2=Activation('relu')(R2_conv2)
    if drop:
        R2_conv2 = Dropout(droprate)(R2_conv2)    
    R2_conv2 = conv2_2(R2_conv2)
    if batch:
        R2_conv2=BatchNormalization(name='R2_conv2_b2')(R2_conv2)
    R2_conv2=Activation('relu')(R2_conv2)
    if drop:
        R2_conv2 = Dropout(droprate)(R2_conv2)
    R2_pool2 = MaxPooling2D(pool_size=(2, 2))(R2_conv2)
##################################################################
    R2_conv3 = conv3_1(R2_pool2)
    if batch:
        R2_conv3=BatchNormalization(name='R2_conv3_b1')(R2_conv3)
    R2_conv3=Activation('relu')(R2_conv3)
    if drop:
        R2_conv3 = Dropout(droprate)(R2_conv3)    
    R2_conv3 = conv3_2(R2_conv3)
    if batch:
        R2_conv3=BatchNormalization(name='R2_conv3_b2')(R2_conv3)
    R2_conv3=Activation('relu')(R2_conv3)
    if drop:
        R2_conv3 = Dropout(droprate)(R2_conv3)
    R2_pool3 = MaxPooling2D(pool_size=(2, 2))(R2_conv3)
##################################################################
    R2_conv4 = conv4_1(R2_pool3)
    if batch:
        R2_conv4=BatchNormalization(name='R2_conv4_b1')(R2_conv4)
    R2_conv4=Activation('relu')(R2_conv4)
    if drop:
        R2_conv4 = Dropout(droprate)(R2_conv4)    
    R2_conv4 = conv4_2(R2_conv4)
    if batch:
        R2_conv4=BatchNormalization(name='R2_conv4_b2')(R2_conv4)
    R2_conv4=Activation('relu')(R2_conv4)
    if drop:
        R2_conv4 = Dropout(droprate)(R2_conv4)
    R2_conv4 = MaxPooling2D(pool_size=(2, 2))(R2_conv4)    
    R2_flat1 = Flatten()(R2_conv4)    
       
      
    
    # FC fusionhead    
    M_flat1 = Concatenate()([R1_flat1, R2_flat1])	
    M_dense1 = Dense(64, kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(wDecay), activity_regularizer=regularizers.l1(aDecay), name='M_dense1')(M_flat1)
    if batch:
        M_dense1=BatchNormalization(name='M_dense1_b1')(M_dense1)
    M_dense1=Activation('relu')(M_dense1)
    if drop:
        M_dense1 = Dropout(droprate)(M_dense1)
    M_dense2 = Dense(nbClasses,activation='softmax',name='M_dense2')(M_dense1)

    
    model = Model(inputs=[inp_1, inp_2], outputs=M_dense2)   
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['acc'])
    return model
