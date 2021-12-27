"""
FABADA is a non-parametric noise reduction technique based on Bayesian
inference that iteratively evaluates possible smoothed  models  of
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
pip install pipwin
pipwin install pyaudio #assuming you're on windows

python thepythonfilename.py #assuming the python file is in the current directory

"""

import numpy
import pyaudio
import scipy.stats
from numpy_ringbuffer import RingBuffer
import numba

def relay (data: [float]):
    data = data.astype(float)
    data = data / 1
    dleft, dright = data[0::2], data[1::2]
    dleft =  fabada1x(dleft)
    dright =  fabada1x(dright)
    data = numpy.concatenate((dleft, dright))
    data2 = numpy.column_stack(numpy.split(data, 2)).ravel().astype(numpy.int16)
    return data2

#(numba.types.Tuple((numba.float64[:],numba.float64[:],numba.float64,numba.float64))(numba.float64[:]))
@numba.jit(nopython=True)
def evaluate(prior_mean: [float],data: [float],prior_variance: float,data_variance: float):
    x = numpy.exp(-((prior_mean - data) ** 2) / (2 * (prior_variance + data_variance))) / numpy.sqrt(
        2 * numpy.pi * (prior_variance + data_variance))
    return x


@numba.jit(numba.types.Tuple((numba.float64[:],numba.float64))( numba.float64[:]))
def variance(data: [float]):
    data = data / 1.0
    data_alpha_padded = numpy.empty(shape=[16384], dtype=float, order='C', like=None)
    data_beta = numpy.empty(shape=[16384], dtype=float, order='C', like=None)
    data_variance_residues = numpy.empty(shape=[16384], dtype=float, order='C', like=None)
    data_variance  = numpy.empty(shape=[16384], dtype=float, order='C', like=None)
     #numpy.ndarray(buffer=data,dtype=float,shape=[16384])
    data_alpha_padded = numpy.concatenate((numpy.full((1,), (data[0] / 2) + (data[1] / 2)), data, numpy.full((1,), (data[-1] / 2) + (data[-2] / 2))))
    # average the data
    data_beta = numpy.asarray([(i + j + k / 3) for i, j, k in
                               zip(data_alpha_padded, data_alpha_padded[1:], data_alpha_padded[2:])])

    # get the smallest positive average, get the smallest out of the two. conveniently this also returns the distance between the average and the not so average
    #which is also known as the variance. ugh! numba why u like this
    #data_variance_residues = [abs(x - j) for x, j in zip(data_beta, data)]
    data_variance_residues = numpy.absolute(data_beta - data)
    #print(data_variance_residues==data_variance_residue2s)
    # we assume beta is larger than residual. after all, few values will be outliers.
    # we want the algorithm to speculatively assume the variance is smaller for data that slopes well per sample.
    variance5 = abs(numpy.var(data_variance_residues))  * 1.61803398875

    data_variance =  data_variance_residues * variance5
    #for some reason sometimes this overflows to NAN, which is a major NONO

    return data_variance,variance5


#@numba.jit(nopython=True) # for some reason this cant devide by zero when compiled
#todo: fix invalid values
def Evidence_start(data_variance: [float]):
    return numpy.exp(-((0 - numpy.sqrt(data_variance)) ** 2) / (2 * (0 + data_variance))) / numpy.sqrt(
        2 * numpy.pi * (0 + data_variance))

def signaltonoise_dB(a, axis=0, ddof=0):
    a = numpy.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*numpy.log10(abs(numpy.where(sd == 0, 0, m/sd)))


def PSNR(recover, signal, L=255):
    MSE = numpy.sum((recover - signal) ** 2) / (recover.size)
    return 10 * numpy.log10((L) ** 2 / MSE)

def fabada1x(data: [float]):
    # fabada expects the data as a floating point array, so, that is what we are going to work with.
    #define the number of iterations based on the SNR of the sample, where noisier samples need more work.
    # the most this can be is around 40 to -80, inverted, adds max of 80, subtracts most of 40.
    max_iter: int = 50 + int(numpy.nan_to_num((signaltonoise_dB(data, axis=0, ddof=0) * -1.0),posinf=80.0,neginf=-40))
    print(max_iter)
    # move buffer calculations
    # Get the channels
    data = data.astype(float)
    # insert the values before and after
    posterior_mean = data
    data_variance ,var5 = variance(data)
    data_variance = numpy.nan_to_num(var5, copy=False)

    posterior_variance = data_variance
    evidence = Evidence_start(data_variance)
    
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
        prior_mean = meanx
        prior_variance = posterior_variance

        # APPLIY BAYES' THEOREM
        posterior_variance = 1 / (1 / prior_variance + 1 / data_variance)
        posterior_mean = (prior_mean / prior_variance + data / data_variance) * posterior_variance

        # EVALUATE EVIDENCE
        evidence = evaluate(prior_mean,data,prior_variance,data_variance)
        evidence_derivative = numpy.mean(evidence) - evidence_previous

        # EVALUATE CHI2
        chi2_data = numpy.sum((data - posterior_mean) ** 2 / data_variance)
        chi2_pdf = scipy.stats.chi2.pdf(chi2_data, df=data.size)
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

    return bayesian_model / bayesian_weight

class StreamSampler(object):

    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.micindex = 1
        self.speakerindex = 1
        self.micstream = self.open_mic_stream()
        self.speakerstream = self.open_speaker_stream()
        self.rb = RingBuffer(capacity=3, dtype=(numpy.int16,32768))

        
    def stop(self):
        self.micstream.close()
        self.speakerstream.close()

    def open_mic_stream(self):
        device_index = None
        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)
            # print("Device %d: %s" % (i, devinfo["name"]))
            if devinfo['maxInputChannels'] == 2:
                for keyword in ["microsoft"]:
                    if keyword in devinfo["name"].lower():
                        print(("Found an input: device %d - %s" % (i, devinfo["name"])))
                        device_index = i
                        self.micindex = device_index

        if device_index is None:
            print("No preferred input found; using default input device.")

        stream = self.pa.open(format=pyaudio.paInt16,
                              channels=2,
                              rate=48000,
                              input=True,
                              input_device_index=self.micindex,  # device_index,
                              frames_per_buffer=16384,
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
                        print(("Found an output: device %d - %s" % (i, devinfo["name"])))
                        device_index = i
                        self.speakerindex = device_index

        if device_index is None:
            print("No preferred output found; using default output device.")

        stream = self.pa.open(format=pyaudio.paInt16,
                              channels=2,
                              rate=48000,
                              output=True,
                              output_device_index=self.speakerindex,
                              frames_per_buffer=16384,
                              stream_callback=self.non_blocking_stream_write,
                              )
        return stream

    # it is critical that this function do as little as possible, as fast as possible. numpy.ndarray is the fastest we can move.
    def non_blocking_stream_read(self, in_data, frame_count, time_info, status):
            self.rb.append(numpy.ndarray(buffer=in_data, dtype=numpy.int16, shape=[32768]))
            return None, pyaudio.paContinue


    def non_blocking_stream_write(self, in_data, frame_count, time_info, status):
            return relay (self.rb[-1]), pyaudio.paContinue
       
        

    def stream_start(self):
        self.micstream.start_stream()
        self.speakerstream.start_stream()
        while self.micstream.is_active():
            eval(input("main thread is now paused"))
        pass

    def listen(self):
        self.stream_start()


if __name__ == "__main__":
    SS = StreamSampler()
    SS.listen()
