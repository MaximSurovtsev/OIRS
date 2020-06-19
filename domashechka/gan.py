#!/usr/bin/env python3
import os
import sys
import numpy as np
import random
from keras import models
from keras import optimizers
from keras.layers import Input
from keras.optimizers import Adam, Adagrad, Adadelta, Adamax, SGD
from keras.callbacks import CSVLogger
import scipy
import h5py
from data import denormalize4gan
from nets import build_discriminator, build_gen, build_enc



def sample_faces(faces):
    reals = []
    for i in range(64) :
        j = random.randrange(len(faces))
        face = faces[j]
        reals.append(face)
    reals = np.array(reals)
    return reals

def sample_fake(gen):
    noise = binary_noise(64)
    fakes = gen.predict(noise)
    return fakes, noise

def binary_noise(cnt):
    noise = 0.1 * np.random.ranf((cnt,) + (1, 1, 100)) 
    noise -= 0.05 
    noise += np.random.randint(0, 2, size=((cnt,) + (1, 1, 100)))

    noise -= 0.5
    noise *= 2
    return noise



def dump_batch(imgs, cnt, ofname):
    assert 64 >= cnt * cnt
    rows = []

    for i in range(cnt) :
        cols = []
        for j in range(cnt*i, cnt*i+cnt):
            cols.append(imgs[j])
        rows.append(np.concatenate(cols, axis=1))

    alles = np.concatenate(rows, axis=0)
    alles = denormalize4gan(alles)
    alles = scipy.misc.imresize(alles, 200) 
    scipy.misc.imsave(ofname, alles)




def build_networks():
    shape = (64, 64, 3)
    dopt = Adam(lr=0.0002, beta_1=0.5)
    opt  = Adam(lr=0.0001, beta_1=0.5)

    gen = build_gen(shape)
    gen.compile(optimizer=opt, loss='binary_crossentropy')
    gen.summary()

    disc = build_discriminator(shape)
    disc.compile(optimizer=dopt, loss='binary_crossentropy')
    disc.summary()

    noise = Input(shape=(1, 1, 100))
    gened = gen(noise)
    result = disc(gened)
    gan = models.Model(inputs=noise, outputs=result)
    gan.compile(optimizer=opt, loss='binary_crossentropy')
    gan.summary()

    return gen, disc, gan




def load_weights(model, wf):
    try:
        model.load_weights(wf)
    except:
        print("failed to load weight, network changed or corrupt hdf5", wf, file=sys.stderr)
        sys.exit(1)




def train_gan(dataf) :
    # Создаем модель
    gen, disc, gan = build_networks()
    logger = CSVLogger('loss.csv') 
    logger.on_train_begin()

    # Запускаем обучение на 500 эпох
    with h5py.File( dataf, 'r' ) as f :
        faces = f.get( 'faces' )
        run_batches(gen, disc, gan, faces, logger, range(5000))
    logger.on_train_end()





def run_batches(gen, disc, gan, faces, logger, itr_generator):
    history = []
    train_disc = True

    for batch in itr_generator:

        lbl_fake = 0.1 * np.random.ranf(64)
        lbl_real = 1 - 0.1 * np.random.ranf(64)

        fakes, noises = sample_fake( gen )
        reals = sample_faces( faces )


        if batch % 10 == 0 :
            if len(history) > 8:
                history.pop(0) 
            history.append( (reals, fakes) )

        # Перестаем обучать генератор
        gen.trainable = False

        d_loss1 = disc.train_on_batch( reals, lbl_real )
        d_loss0 = disc.train_on_batch( fakes, lbl_fake )

        gen.trainable = True

        if batch < 20:
            print( batch, "d0:{} d1:{}".format( d_loss0, d_loss1 ) )
            continue

        disc.trainable = False
        g_loss = gan.train_on_batch( noises, lbl_real )
        disc.trainable = True

        print( batch, "d0:{} d1:{}   g:{}".format( d_loss0, d_loss1, g_loss ) )

        # Сохранение весов каждые 10 эпох
        if batch % 10 == 0 and batch != 0 :
            end_of_batch_task(batch, gen, disc, reals, fakes)
            row = {"d_loss0": d_loss0, "d_loss1": d_loss1, "g_loss": g_loss}
            logger.on_epoch_end(batch, row)



_bits = binary_noise(64)

def end_of_batch_task(batch, gen, disc, reals, fakes):
    try :

        frame = gen.predict(_bits)
        animf = os.path.join("anim", "frame_{:08d}.png".format(int(batch/10)))
        dump_batch(frame, 4, animf)

        serial = int(batch / 10) % 10
        prefix = os.path.join('./snapshots/', str(serial) + ".")

        print("Saving weights", serial)
        gen.save_weights(prefix + 'gen.hdf5')
        disc.save_weights(prefix + 'dics.hdf5')

    except KeyboardInterrupt :
        end_of_batch_task(batch, gen, disc, reals, fakes)
        raise



def generate(cnt):
    shape = (64, 64, 3)
    gen = build_gen(shape)
    gen.compile(optimizer='sgd', loss='mse')
    load_weights(gen, 'snapshots/9.gen.hdf5')

    for i in range(1, 21):
        generated = gen.predict(binary_noise(64))

        animf = os.path.join('GENERATED', "КОРТИНОЧКА{}.png".format(i))
        dump_batch(generated, 8, animf)



if __name__ == '__main__':
    #train_gan("data.hdf5")
    generate(10)







