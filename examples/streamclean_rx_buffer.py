"""
FABADA is a non-parametric noise reduction technique based on Bayesian
inference that iteratively evaluates possibles moothed  models  of
the  data introduced,  obtaining  an  estimation  of the  underlying
signal that is statistically  compatible  with the  noisy  measurements.

based on P.M. Sanchez-Alarcon, Y. Ascasibar, 2022
"Fully Adaptive Bayesian Algorithm for Data Analysis. FABADA"

Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
Everyone is permitted to copy and distribute verbatim copies
of this license document, but changing it is not allowed.


Instructions:
Save the code as a .py file.
Install the latest miniforge for you into a folder, don't add it to path, launch it from start menu.
Note: if python is installed elsewhere this may fail. If it fails, try this again with miniconda instead,
as miniconda doesn't install packages to the system library locations.

https://github.com/conda-forge/miniforge/#download

https://docs.conda.io/en/latest/miniconda.html
(using miniforge command line window)
conda install numba, scipy, numpy, pipwin
pip install pipwin, np_rw_buffer
pipwin install pyaudio

python thepythonfilename.py #assuming the python file is in the current directory

"""

import pyaudio
from scipy.stats import chi2
from numba import jit, types
import np_rw_buffer
import numpy

@jit(types.float64[:](types.float64[:]))
def variance(data: [float]):
    # establish baseline
    data_alpha = data
    # get an array of the mean
    # establish buffer values for x/y/z so that I can get accurate averages for first and last elements.
    # to be able to perform x+y+z/3 on all original elements, we need the first and last to work.
    # to do that, we have to append a new value to both ends which won't distort the original.
    # algebraicly, this is x/2 + y/2, where X and Y are the last and next to last of the original array.
    avar = numpy.full((1,),(data_alpha[0] / 2) + (data_alpha[1] / 2))
    zvar = numpy.full((1,),(data_alpha[-1] / 2) + (data_alpha[-2] / 2))
    # insert the values before and after
    data_alpha_padded = numpy.concatenate((avar, data_alpha,zvar))
    # average the data
    data_beta = numpy.asarray([((i) + (j) + (k) / 3) for i, j, k in
                 zip(data_alpha_padded, data_alpha_padded[1:], data_alpha_padded[2:])])
    # get an array filled with the mean
    x9 =  numpy.mean(data_beta)
    data_mean_beta = numpy.full((16384,), x9)
    # get the variance for each element from the mean
    data_variance_beta = [abs(i-j) for i, j in zip(data_beta, data_mean_beta)]

    # subtract the averages from the original
    data_residues = [i - j for i, j in zip(data_alpha, data_beta)]

    # if subtracting the average would change the sign, leave the original
   # data_residues = numpy.where(numpy.sign(data_alpha) != numpy.sign(data_residues), data_alpha, data_beta)
    data_residues = numpy.asarray([x if j + 0 - j == 0 and x + 0 - x == 0 else j for j,x in zip(data_alpha,data_beta)])
    # get the mean for the residuals left over and/or the original values
    x10 = numpy.mean(data_residues)
    data_mean_residues = numpy.full((16384,),x10)
    # get the variance for the residue forms
    data_variance_residues = numpy.asarray([abs(i-j) for i, j in zip(data_residues, data_mean_residues)])
    # we assume beta is larger than residual.
    # we want the algorithm to speculatively assume the variance is smaller for data that slopes well per sample.
    # after all, noise is very small in the time domain. Man-made noise isn't, but that's a different problem..
    minimum = 1498258522.00 # in my experience this is the minimum amount of noise present
    variance5 = numpy.var(data_variance_residues)
    data_variance_minimum = numpy.full((16384,), minimum)
    data = [x * variance5 for x in data_variance_residues]
    #keep the value from going below the noise barrier. We generally see more noise than this.
    data  = numpy.asarray([max(i,j) for i, j in zip(data, data_variance_minimum)])

    return data

#@jit(types.float64[:](types.float64[:]))
def fabada1x(data: [float]):
    # fabada expects the data as a floating point array, so, that is what we are going to work with.
    # however, attempting to feed it floating point input seems to slow it way down.
    max_iter: int = 128
    #move buffer calculations
    data = data.flatten() #numpy.flatten(data)#ravel(data)
    # Get the channels

    dleft, dright = data[0::2], data[1::2]
    # concat the channel samples as two separate arrays. Remember to reverse this before the end!
    data = numpy.concatenate((dleft, dright))


    # convert to floating point values edit: for numba, move this out of code
    #data = numpy.array(data,dtype=float)
    # copy data

    data_variance = variance(data)
    posterior_mean = data
    posterior_variance = data_variance
    evidence =  numpy.exp(-((0 - numpy.sqrt(data_variance)) ** 2) / (2 * (0 + data_variance))) / numpy.sqrt(
               2 * numpy.pi * (0 + data_variance)
         )
    initial_evidence = evidence
    chi2_pdf, chi2_data, iteration = 0, data.size, 0
    chi2_pdf_derivative, chi2_data_min = 0, data.size
    bayesian_weight = 0
    bayesian_model = 0

    converged = False

    while not converged:

            chi2_pdf_previous = chi2_pdf
            chi2_pdf_derivative_previous = chi2_pdf_derivative
            evidence_previous = numpy.mean(evidence)

            iteration += 1  # Check number of iterations done

            # GENERATES PRIORS
            meanx = posterior_mean.copy()
            meanx[:-1] += posterior_mean[1:]
            meanx[1:] += posterior_mean[:-1]
            meanx[1:-1] /= 3
            meanx[0] /= 2
            meanx[-1] /= 2
            prior_mean  = meanx
            prior_variance = posterior_variance

            # APPLIY BAYES' THEOREM
            posterior_variance = 1 / (1 / prior_variance + 1 / data_variance)
            posterior_mean = (prior_mean / prior_variance + data / data_variance) * posterior_variance


            # EVALUATE EVIDENCE
            evidence = numpy.exp(-((prior_mean - data) ** 2) / (2 * (prior_variance + data_variance))) / numpy.sqrt(
                2 * numpy.pi * (prior_variance + data_variance)
            )

            evidence_derivative = numpy.mean(evidence) - evidence_previous

            # EVALUATE CHI2
            chi2_data = numpy.sum((data - posterior_mean) ** 2 / data_variance)
            chi2_pdf = chi2.pdf(chi2_data, df=data.size)
            chi2_pdf_derivative = chi2_pdf - chi2_pdf_previous
            chi2_pdf_snd_derivative = chi2_pdf_derivative - chi2_pdf_derivative_previous

            # COMBINE MODELS FOR THE ESTIMATION
            model_weight = evidence * chi2_data
            bayesian_weight += model_weight
            bayesian_model += model_weight * posterior_mean

            if iteration == 1:
                chi2_data_min = chi2_data
            # CHECK CONVERGENCE
            if (
                    (chi2_data > data.size and chi2_pdf_snd_derivative >= 0)
                    and (evidence_derivative < 0)
                    or (iteration > max_iter)
            ):
                converged = True

                # COMBINE ITERATION ZERO
                model_weight = initial_evidence * chi2_data_min
                bayesian_weight += model_weight
                bayesian_model += model_weight * data

    bayes = bayesian_model / bayesian_weight
    # recombine the channels into one interleaved set of samples
    data2 = numpy.column_stack(numpy.split(bayes, 2)).ravel().astype(numpy.int16)
    return data2


#@jit(nopython=True)
#def PSNR(recover, signal, L=255):
 #   MSE = numpy.sum((recover - signal) ** 2) / (recover.size)
 #   return 10 * numpy.log10((L) ** 2 / MSE)


class StreamSampler(object):

    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.micindex = 1
        self.speakerindex = 1
        self.buffer = np_rw_buffer.RingBuffer(16384 * 4, 0, numpy.int16)
        self.micstream = self.open_mic_stream()
        self.speakerstream = self.open_speaker_stream()
        self.errorcount = 0
        self.previousvariance = 0.0

    def stop(self):
        self.micstream.close()
        self.speakerstream.close()
        self.pool.shutdown(block=False)

    def open_mic_stream(self):
        device_index = None
        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)
            # print("Device %d: %s" % (i, devinfo["name"]))
            if devinfo['maxInputChannels'] == 2:
                for keyword in ["microsoft"]:
                    if keyword in devinfo["name"].lower():
                        print("Found an input: device %d - %s" % (i, devinfo["name"]))
                        device_index = i
                        self.micindex = device_index

        if device_index == None:
            print("No preferred input found; using default input device.")

        stream = self.pa.open(format=pyaudio.paInt16,
                              channels=2,
                              rate=48000,
                              input=True,
                              input_device_index=self.micindex,  # device_index,
                              frames_per_buffer=8192,
                              stream_callback=self.non_blocking_stream_read,
                              )

        return stream

    def open_speaker_stream(self):
        device_index = None
        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)
            # print("Device %d: %s" % (i, devinfo["name"]))
            if devinfo['maxOutputChannels'] == 2:
                for keyword in ["microsoft"]:
                    if keyword in devinfo["name"].lower():
                        print("Found an output: device %d - %s" % (i, devinfo["name"]))
                        device_index = i
                        self.speakerindex = device_index

        if device_index == None:
            print("No preferred output found; using default input device.")

        stream = self.pa.open(format=pyaudio.paInt16,
                              channels=2,
                              rate=48000,
                              output=True,
                              output_device_index=self.speakerindex,
                              frames_per_buffer=8192,
                              stream_callback=self.non_blocking_stream_write,
                              )
        return stream
    def non_blocking_stream_read(self, in_data, frame_count, time_info, status):
        return self.buffer.write_value(numpy.frombuffer(in_data, count=16384, dtype=numpy.int16).reshape(-1, 1) ,
                                     length=16384, error=False), pyaudio.paContinue

    def non_blocking_stream_write(self, in_data, frame_count, time_info, status):
        print(" got to here ")
        return fabada1x(numpy.asarray(self.buffer.read_overlap(amount=16384, increment=16384),
                                      dtype=float)), pyaudio.paContinue

    def stream_start(self):
        self.micstream.start_stream()
        self.speakerstream.start_stream()
        while self.micstream.is_active():
            input("main thread is now paused")
        return

    def listen(self):
        self.stream_start()


if __name__ == "__main__":
    SS = StreamSampler()
    SS.listen()
