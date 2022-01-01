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
conda install numba, scipy, numpy, pipwin, np_rw_buffer
pip install pipwin
pipwin install pyaudio #assuming you're on windows

python thepythonfilename.py #assuming the python file is in the current directory

Usage:
You'll need a line-in device or virtual audio cable you can configure so you can loop the output to input.
The streaming example included here looks for the windows muxer that configures audio devices- whatever you set
in your windows settings for default mic and speaker, respectively, this program will treat as input and output.
So, you can configure another program to output noisy sound to the speaker side of a virtual audio device, and configure
the microphone end of that device as your system microphone, then this program will automatically pick it up and run it.
https://vb-audio.com/Cable/ is an example of a free audio cable.
The program expects 48000hz audio, 16 bit, two channel, but can be configured to work with anything thanks to Justin Engel.

"""
import numpy
import pyaudio
import numba
from np_rw_buffer import AudioFramingBuffer
from threading import Thread
import math
import time

TAU: numpy.float64 = 2 * numpy.pi


@numba.jit ((numba.float64)(numba.float64[:]))
def signaltonoise_dB(data: [numpy.float64]):
        m = data.mean()
        sd = data.std()
        if sd == 0:
            xl = 0
        else:
            xl = m/sd
        xd = abs(xl)
        return 20*numpy.log10(xd)

@numba.jit (numba.float64[:](numba.float64[:],numba.float64[:],numba.float64[:],numba.float64[:]))
def evaluate(prior_mean: [numpy.float64],data: [numpy.float64],prior_variance: numpy.float64,data_variance: numpy.float64):
        return numpy.exp(-((prior_mean - data) ** 2) / (2 * (prior_variance + data_variance))) / numpy.sqrt( TAU * (prior_variance + data_variance))


#evidence = Evidence(0, numpy.sqrt(data_variance), 0, data_variance)
#Evidence(prior_mean, data, prior_variance, data_variance)
#properly specifying numba signature fixes this
@numba.jit (numba.float64[:](numba.float64[:],numba.float64[:],numba.float64[:],numba.float64[:]))
def Evidence(mu1: [numpy.float64], mu2: [numpy.float64], var1: [numpy.float64], var2: [numpy.float64]):
    return numpy.exp(-((mu1 - mu2) ** 2) / (2 * (var1 + var2))) / numpy.sqrt( TAU * (var1 + var2))


#first time this function is called, it's unfortunantly called with 0.0, which is not useful here
@numba.jit (numba.float64[:](numba.float64[:]))
def Evidence1st(var2: [numpy.float64]):
    return numpy.exp(-((0.0 - numpy.sqrt(var2)) ** 2) / (2 * (0.0 + var2))) / numpy.sqrt(2 * numpy.pi * (0.0 + var2))


#
#prespecifying the nuba types fixed this tremendously, it no longer takes 3000 milliseconds the first time it runs
@numba.jit((numba.float64[:])(numba.float64[:]))
def meanx1(data: [numpy.float64]):
    meanx:[numpy.float64] = data / 1.0
    meanx[:-1] += data[1:]
    meanx[1:] += data[:-1]
    meanx[1:-1] /= 3
    meanx[0] /= 2
    meanx[-1] /= 2
    return meanx
    #simple, easy, 1-dimensional function

@numba.jit#((numba.float64[::])(numba.float64[::]))
def meanx2(data:[numpy.float64]):
    mean:[numpy.float64] = data / 1.0
    mean[:-1, :] += data[1:, :]
    mean[1:, :] += data[:-1, :]
    mean[:, :-1] += data[:, 1:]
    mean[:, 1:] += data[:, :-1]
    mean[1:-1, 1:-1] /= 5
    mean[0, 1:-1] /= 4
    mean[-1, 1:-1] /= 4
    mean[1:-1, 0] /= 4
    mean[1:-1, -1] /= 4
    mean[0, 0] /= 3
    mean[-1, -1] /= 3
    mean[0, -1] /= 3
    mean[-1, 0] /= 3
    return mean #simple, easy, 2-dimensional function


@numba.jit(numba.float64[:](numba.float64[:],numba.float64[:],numba.float64[:],numba.float64[:],numba.float64[:]))
def posterior_mean_gen(prior_mean: [numpy.float64],prior_variance: [numpy.float64],data: [numpy.float64],data_variance: [numpy.float64],posterior_variance: [numpy.float64]):
    return ( prior_mean / prior_variance + data / data_variance ) * posterior_variance

@numba.jit(numba.float64(numba.float64))
def chi2_pdf_call(x: numpy.float64):
    df: int = 28 #note: this isnt the right way to use this function. DF is supposed to be, like data.size but that would be enormous
    ## chi2.pdf(x, df) = 1 / (2*gamma(df/2)) * (x/2)**(df/2-1) * exp(-x/2)
    gammar: numpy.float64 = (2. * math.lgamma(df / 2.))
    gammaz: numpy.float64 = ((df / 2.) - 1.)
    gamman: numpy.float64 = (x / 2.)
    gammas: numpy.float64 = (numpy.sign(gamman) * ((numpy.abs(gamman)) ** gammaz)) #raising this to the powa will just result in FUBAR
    gammaq: numpy.float64 =  numpy.exp(-x / 2.)
    gammaa: numpy.float64 =  1. / gammar
    pdf: numpy.float64 = gammaa * gammas * gammaq
    return pdf
    #this is the correct, optimized chi2_pdf function call. There is nothing wrong in this code.
    #This code returns the exact equivalent of calling scipy.stats.chi2.pdf(x,df)
    #but, where scipy code can't be optimized with numba, this can.
    #however, it doesnt do you any good if you're trying to raise 22.9999 to the 47,999 power
    #xlogy     -- Compute ``x*log(y)`` so that the result is 0 if ``x = 0``. https://github.com/scipy/scipy/blob/master/scipy/special/_xlogy.pxd
    #gammaln      -- Logarithm of the absolute value of the Gamma function for real inputs. same thing as math.lgamma. https://github.com/scipy/scipy/blob/701ffcc8a6f04509d115aac5e5681c538b5265a2/scipy/special/cephes/gamma.c




@numba.jit((numba.float64[:])( numba.float64[:]))
def nan(data: [numpy.float64]):
    return numpy.asarray([x if not math.isnan(x) else 0.0 for x in data],dtype=numpy.float64)

@numba.jit(numba.int16( numba.int16,numba.int16,numba.int16))
def bound(value: int, low: int = 20, high: int =100):
     diff = high - low
     return (((value - low) % diff) + low)

@numba.jit((numba.float64[:])( numba.float64[:]))
def variance(data: [numpy.float64]):

    #note: nothing in this function is set in stone or even good.
    #the contents of this function determine how fabada sees the need for convolution.
    #i guess. i dont really know. But this is the source of the clicking, the dropped samples, etc.
    #the only time fabada will work for this particular application is when we get this right.
    data_mean = abs(numpy.median(data))* numpy.ones_like(data,dtype=numpy.float64)  # get the mean
    # The formula for standard deviation is the square root of the sum of squared differences from the mean divided by the size of the data set.
    data_variance: [numpy.float64] = numpy.asarray([(abs(j - x)) for j, x in zip(data_mean,data)])
    data_variance = data_variance + 1024.0 # bring le floor up- whatever this is truncates higher frequency data.
    #64 seems good, 92 seems strong enough
    #if we bring this up by 0, the output is extremely noisy.
    data_variance= (data_variance ** 2 ) #exponentiate
    return data_variance

@numba.jit((numba.float64[:])( numba.float64[:],numba.float64,numba.float64, numba.float64,numba.float64))
def interpolate(data:[numpy.float64],min_d: numpy.float64,max_d: numpy.float64,zoop: numpy.float64,zoo: numpy.float64):
    return numpy.interp(data, (min_d, max_d), (zoop, zoo))

@numba.jit((numba.float64)( numba.float64[:],numba.float64[:],numba.float64[:]))
def power(data:[numpy.float64],posterior_mean: [numpy.float64],data_variance: [numpy.float64]):
    #twos = 2.0 * numpy.ones_like(data)
    return numpy.sum((data - posterior_mean) ** 2 / data_variance)

@numba.jit(numba.float64[:](numba.float64[:],numba.float64[:]))
def postisum(prior_variance : [numpy.float64],data_variance: [numpy.float64]):
    return 1.0 / (1.0 / prior_variance + 1.0 / data_variance)

def numba_fabada(data: [numpy.float64]):
        start = time.time()
        bayesian_weight: [numpy.float64] = numpy.zeros_like(data)
        bayesian_model:  [numpy.float64] = numpy.zeros_like(data)
        max_iter: int = 3000
        #this has to run as much as possible, but unfortunantly, it takes a looong time to run on this much data.

        #numba doesn't like unions, arbitrary logic, so for fabada to work on 2x the parent thread needs to call
        #a different function, and internally a different means and variance calculation function.
        #This work is beyond the scope of this author

        # data_variance = numpy.array(data_variance / 1.0)
        min_d: numpy.float64= numpy.min(data)
        max_d: numpy.float64= numpy.ptp(data)
        min: numpy.float64 = 0.0
        max: numpy.float64 = 255.0
        #normalize the datum
        data: [numpy.float64] =  interpolate(data, min_d, max_d,min, max)

        data_variance: [numpy.float64] = variance(data)




        # initialize bayes for the function return

        # INITIALIZING ALGORITMH ITERATION ZERO

        posterior_mean: [numpy.float64] = data
        posterior_variance: [numpy.float64] = data_variance

        evidence: [numpy.float64] = Evidence1st(data_variance)
        initial_evidence: [numpy.float64] = evidence
        iteration: int = 0.0
        chi2_pdf: numpy.float64 = 0.0
        chi2_pdf_derivative: numpy.float64 = 0.0

        # converged = False
        iteration += 1  # set  number of iterations done

        chi2_pdf_previous: numpy.float64 = chi2_pdf
        chi2_pdf_derivative_previous = chi2_pdf_derivative
        evidence_previous: [numpy.float64] = numpy.mean(evidence)

        # GENERATES PRIORS

        prior_mean: [numpy.float64] = meanx1(posterior_mean)

        prior_variance: [numpy.float64] = posterior_variance

        # APPLIY BAYES' THEOREM

        posterior_variance: [numpy.float64] =  postisum(prior_variance,data_variance)
        posterior_mean: [numpy.float64] = posterior_mean_gen(prior_mean, prior_variance, data, data_variance, posterior_variance)
        # EVALUATE EVIDENCE
        evidence: [numpy.float64] = Evidence(prior_mean, data, prior_variance, data_variance)
        evidence_derivative: [numpy.float64] = numpy.subtract(numpy.mean(evidence), evidence_previous)

        # EVALUATE CHI2
        #numpy does not allow fractional powers of negative numbers!

        chi2_data = power(data, posterior_mean,data_variance)
        chi2_pdf: numpy.float64 = chi2_pdf_call(chi2_data)
        chi2_pdf_derivative = chi2_pdf - chi2_pdf_previous
        chi2_pdf_snd_derivative = chi2_pdf_derivative - chi2_pdf_derivative_previous


        # COMBINE MODELS FOR THE ESTIMATION
        model_weight: [numpy.float64] = numpy.multiply(evidence, chi2_data)
        bayesian_weight: [numpy.float64] = numpy.add(bayesian_weight, model_weight)
        bayesian_model: [numpy.float64] = numpy.add(bayesian_model, numpy.multiply(model_weight, posterior_mean))

        chi2_data_min: numpy.float64 = chi2_data

        while 1:
            current = time.time()
            timerun = (current - start) * 1000
            if (
                    (int(chi2_data) > data.size and chi2_pdf_snd_derivative >= 0)
                    or (evidence_derivative < 0)
                    or (iteration > max_iter)
                    or (timerun > 900)#use no more than 90% of the time allocated per cycle

            ):
                break
                # break the loop when we're done by prematurely setting converged and returning
                # this allows us to test if one round of convergence is enough.
            # else:# the else here is redundant, as we either do or dont do, continue resets the while
            # and break terminates. While 1 is the fastest we can operate in this mode

            chi2_pdf_previous: numpy.float64 = chi2_pdf
            chi2_pdf_derivative_previous: numpy.float64 = chi2_pdf_derivative
            evidence_previous: [numpy.float64] = numpy.mean(evidence,dtype=numpy.float64)

            iteration += 1  # Check number of iterartions done

            # GENERATES PRIORS
            prior_mean: [numpy.float64] = meanx1(posterior_mean)
            # if(posterior_mean.ndim == 2):
            #    prior_mean = meanx2(posterior_mean)
            prior_variance: [numpy.float64] = posterior_variance

            # APPLIY BAYES' THEOREM
            # prevent le' devide by le zeros
            posterior_variance: [numpy.float64] = postisum(prior_variance,data_variance)

            posterior_mean: [numpy.float64] = posterior_mean_gen(prior_mean, prior_variance, data, data_variance, posterior_variance)

            # EVALUATE EVIDENCE
            evidence: [numpy.float64] = Evidence(prior_mean, data, prior_variance, data_variance)
            evidence_derivative: [numpy.float64] = numpy.subtract(numpy.mean(evidence), evidence_previous)

            # EVALUATE CHI2
            chi2_data: [numpy.float64]  = power(data, posterior_mean,data_variance)

            chi2_pdf: numpy.float64 = chi2_pdf_call(chi2_data)
            chi2_pdf_derivative: numpy.float64 = chi2_pdf - chi2_pdf_previous
            chi2_pdf_snd_derivative: numpy.float64 = chi2_pdf_derivative - chi2_pdf_derivative_previous

            # COMBINE MODELS FOR THE ESTIMATION
            model_weight: [numpy.float64] = numpy.multiply(evidence,chi2_data)
            bayesian_weight: [numpy.float64] = numpy.add(bayesian_weight, model_weight)

            bayesian_model: [numpy.float64] = numpy.add(bayesian_model, numpy.multiply(model_weight,posterior_mean))


            # COMBINE ITERATION ZERO
        model_weight: [numpy.float64] = numpy.multiply(initial_evidence, chi2_data_min)
        bayesian_weight: [numpy.float64] = numpy.add(model_weight,bayesian_weight)
        bayesian_model: [numpy.float64] = numpy.add(bayesian_model, numpy.multiply(model_weight, data))

                #de-normalize the datum
        output: [numpy.float64] =  interpolate(numpy.divide(bayesian_model,bayesian_weight),min, max, min_d, +max_d)
        return output


class FilterRun(Thread):
    def __init__(self,rb,pb,channels,processing_size,dtype):
        super(FilterRun, self).__init__()
        self.running = True
        self.rb = rb
        self.processedrb = pb
        self.channels = channels
        self.processing_size = processing_size
        self.dtype = dtype
        self.buffer = numpy.ndarray(dtype=numpy.float64, shape=[int(self.processing_size * self.channels)])
        self.buffer = self.buffer.reshape(-1,self.channels)

    def write_filtered_data(self):
        audio = self.rb.read(self.processing_size).astype(dtype=numpy.float64)
        for i in range(self.channels):
            self.buffer[:, i]  = numba_fabada(audio[:, i])
        self.processedrb.write(self.buffer.astype(dtype=self.dtype),error=True)

    def run(self):
        while self.running:
            if len(self.rb) < self.processing_size * 2:
                time.sleep(0.001)
            else:
                self.write_filtered_data()


    def stop(self):
        self.running = False



class StreamSampler(object):

    dtype_to_paformat = {
        # Numpy dtype : pyaudio enum
        'uint8': pyaudio.paUInt8,
        'int8': pyaudio.paInt8,
        'uint16': pyaudio.paInt16,
        'int16': pyaudio.paInt16,
        'uint24': pyaudio.paInt24,
        'int24': pyaudio.paInt24,
        "uint32": pyaudio.paInt32,
        'int32': pyaudio.paInt32,
        'float32': pyaudio.paFloat32,

        # Float64 is not a valid pyaudio type.
        # The encode method changes this to a float32 before sending to audio
        'float64': pyaudio.paFloat32,
        "complex128": pyaudio.paFloat32,
        }

    @classmethod
    def get_pa_format(cls, dtype):
        try:
            dtype = dtype.dtype
        except (AttributeError, Exception):
            pass
        return cls.dtype_to_paformat[dtype.name]


    def __init__(self, sample_rate=44100, channels=2, buffer_delay=1.5, # or 1.5, measured in seconds
                 micindex=1, speakerindex=1, dtype=numpy.int32):
        self.pa = pyaudio.PyAudio()
        self._processing_size = int(sample_rate * 2)
        # np_rw_buffer (AudioFramingBuffer offers a delay time)
        self._sample_rate = sample_rate
        self._channels = channels
        # self.rb = RingBuffer((int(sample_rate) * 5, channels), dtype=numpy.dtype(dtype))
        self.rb = AudioFramingBuffer(sample_rate=sample_rate, channels=channels,
                                     seconds=6,  # Buffer size (need larger than processing size)[seconds * sample_rate]
                                     buffer_delay=0,  # #this buffer doesnt need to have a size
                                     dtype=numpy.dtype(dtype))

        self.processedrb = AudioFramingBuffer(sample_rate=sample_rate, channels=channels,
                                     seconds=6,  # Buffer size (need larger than processing size)[seconds * sample_rate]
                                     buffer_delay=1, # as long as fabada completes in O(n) of less than the sample size in time
                                     dtype=numpy.dtype(dtype))

        self.filterthread = FilterRun(self.rb,self.processedrb,self._channels,self._processing_size,self.dtype)
        self.micindex = micindex
        self.speakerindex = speakerindex
        self.micstream = None
        self.speakerstream = None

        # Set inputs for inheritance
        self.set_sample_rate(sample_rate)
        self.set_channels(channels)
        self.set_dtype(dtype)

    @property
    def processing_size(self):
        return self._processing_size

    @processing_size.setter
    def processing_size(self, value):
        self._processing_size = value
        self._update_streams()

    def get_sample_rate(self):
        return self._sample_rate

    def set_sample_rate(self, value):
        self._sample_rate = value
        try:  # RingBuffer
            self.rb.maxsize = int(value * 5)
            self.processedrb.maxsize = int(value * 5)
        except AttributeError:
            pass
        try:  # AudioFramingBuffer
            self.rb.sample_rate = value
            self.processedrb.sample_rate = value
        except AttributeError:
            pass
        self._update_streams()

    sample_rate = property(get_sample_rate, set_sample_rate)

    def get_channels(self):
        return self._channels

    def set_channels(self, value):
        self._channels = value
        try:  # RingBuffer
            self.rb.columns = value
            self.processedrb.columns = value
        except AttributeError:
            pass
        try:  # AudioFrammingBuffer
            self.rb.channels = value
            self.processedrb.channels = value
        except AttributeError:
            pass
        self._update_streams()

    channels = property(get_channels, set_channels)

    def get_dtype(self):
        return self.rb.dtype

    def set_dtype(self, value):
        try:
            self.rb.dtype = value
        except AttributeError:
            pass
        self._update_streams()

    dtype = property(get_dtype, set_dtype)

    @property
    def pa_format(self):
        return self.get_pa_format(self.dtype)

    @pa_format.setter
    def pa_format(self, value):
        for np_dtype, pa_fmt in self.dtype_to_paformat.items():
            if value == pa_fmt:
                self.dtype = numpy.dtype(np_dtype)
                return

        raise ValueError('Invalid pyaudio format given!')

    @property
    def buffer_delay(self):
        try:
            return self.rb.buffer_delay
        except (AttributeError, Exception):
            return 0

    @buffer_delay.setter
    def buffer_delay(self, value):
        try:
            self.rb.buffer_delay = value
            self.processedrb.buffer_delay = value
        except AttributeError:
            pass

    def _update_streams(self):
        """Call if sample rate, channels, dtype, or something about the stream changes."""
        was_running = self.is_running()

        self.stop()
        self.micstream = None
        self.speakerstream = None
        if was_running:
            self.listen()

    def is_running(self):
        try:
            return self.micstream.is_active() or self.speakerstream.is_active()
        except (AttributeError, Exception):
            return False

    def stop(self):
        try:
            self.micstream.close()
        except (AttributeError, Exception):
            pass
        try:
            self.speakerstream.close()
        except (AttributeError, Exception):
            pass
        try:
            self.filterthread.join()
        except (AttributeError, Exception):
            pass

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

        stream = self.pa.open(format=self.pa_format,
                              channels=self.channels,
                              rate=int(self.sample_rate),
                              input=True,
                              input_device_index=self.micindex,  # device_index,
                              #each frame carries twice the data of the frames
                              frames_per_buffer= int(self._processing_size),
                              stream_callback=self.non_blocking_stream_read,
                              start=False  # Need start to be False if you don't want this to start right away
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

        stream = self.pa.open(format=self.pa_format,
                              channels=self.channels,
                              rate=int(self.sample_rate),
                              output=True,
                              output_device_index=self.speakerindex,
                              frames_per_buffer= int(self._processing_size),
                              stream_callback=self.non_blocking_stream_write,
                              start=False  # Need start to be False if you don't want this to start right away
                              )
        return stream

    # it is critical that this function do as little as possible, as fast as possible. numpy.ndarray is the fastest we can move.
    # attention: numpy.ndarray is actually faster than frombuffer for known buffer sizes
    def non_blocking_stream_read(self, in_data, frame_count, time_info, status):
        audio_in = memoryview(numpy.ndarray(buffer=memoryview(in_data), dtype=self.dtype, shape=[int(self._processing_size* self._channels)]).reshape(-1, self.channels))
        self.rb.write(audio_in, error=False)
        return None, pyaudio.paContinue

    def non_blocking_stream_write(self, in_data, frame_count, time_info, status):
        # Read raw data
        # filtered = self.rb.read(frame_count)
        # if len(filtered) < frame_count:
        #     filtered = numpy.zeros((frame_count, self.channels), dtype=self.dtype)
        if len(self.processedrb) < self.processing_size:
            #print('Not enough data to play! Increase the buffer_delay')
            #uncomment this for debug
            audio = numpy.zeros((self.processing_size, self.channels), dtype=self.dtype)
            return audio, pyaudio.paContinue

        audio = self.processedrb.read(self.processing_size)
        chans = []
        for i in range(self.channels):
            filtered = audio[:, i]
            chans.append(filtered)

        return numpy.column_stack(chans).astype(self.dtype).tobytes(), pyaudio.paContinue

    def stream_start(self):
        if self.micstream is None:
            self.micstream = self.open_mic_stream()
        self.micstream.start_stream()

        if self.speakerstream is None:
            self.speakerstream = self.open_speaker_stream()
        self.speakerstream.start_stream()
        # Don't do this here. Do it in main. Other things may want to run while this stream is running
        # while self.micstream.is_active():
        #     eval(input("main thread is now paused"))

    listen = stream_start  # Just set a new variable to the same method


if __name__ == "__main__":
    SS = StreamSampler(buffer_delay=0)
    SS.listen()
    SS.filterthread.start()

    while SS.is_running():
        inp = input('Press enter to quit!\n')   # Halt until user input
        SS.filterthread.stop()
        SS.stop()
        break

    #SS.stop()
