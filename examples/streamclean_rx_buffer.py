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
import scipy
import scipy.stats
import numba
from np_rw_buffer import AudioFramingBuffer
import sys
from threading import Thread
import time
import math
from time import time as timer

@numba.jit ((numba.float64)(numba.float64[:]))
def signaltonoise_dB(data: [float]):
        data = data / 1.0
        m = data.mean()
        sd = data.std()
        if sd == 0:
            xl = 0
        else:
            xl = m/sd
        xd = abs(xl)
        return 20*numpy.log10(xd)

@numba.jit (numba.float64[:](numba.float64[:],numba.float64[:],numba.float64[:],numba.float64[:]))
def evaluate(prior_mean: [float],data: [float],prior_variance: float,data_variance: float):
        x = numpy.exp(-((prior_mean - data) ** 2) / (2 * (prior_variance + data_variance))) / numpy.sqrt(
            2 * numpy.pi * (prior_variance + data_variance))
        return x

#evidence = Evidence(0, numpy.sqrt(data_variance), 0, data_variance)
#Evidence(prior_mean, data, prior_variance, data_variance)
#properly specifying numba signature fixes this
@numba.jit (numba.float64[:](numba.float64[:],numba.float64[:],numba.float64[:],numba.float64[:]))
def Evidence(mu1: [float], mu2: [float], var1: [float], var2: [float]):
    return numpy.exp(-((mu1 - mu2) ** 2) / (2 * (var1 + var2))) / numpy.sqrt(
        2 * numpy.pi * (var1 + var2)
    )

#first time this function is called, it's unfortunantly called with 0.0, which is not useful here
@numba.jit (numba.float64[:](numba.float64,numba.float64[:],numba.float64,numba.float64[:]))
def Evidence1st(mu1: float, mu2: [float], var1: float, var2: [float]):
    return numpy.exp(-((mu1 - mu2) ** 2) / (2 * (var1 + var2))) / numpy.sqrt(
        2 * numpy.pi * (var1 + var2)
    )

#
#prespecifying the nuba types fixed this tremendously, it no longer takes 3000 milliseconds the first time it runs
@numba.jit((numba.float64[:])(numba.float64[:]))
def meanx1(data: [float]):
    meanx = data / 1.0
    meanx[:-1] += data[1:]
    meanx[1:] += data[:-1]
    meanx[1:-1] /= 3
    meanx[0] /= 2
    meanx[-1] /= 2
    return meanx
    #simple, easy, 1-dimensional function

#this code works but it doesnt work very well.
#as an example of how to vectorize code, it works very well, but vectorizing this 1d transform doesn't work well.
#on the other hand, maybe i just wrote it wrong? something to do with the way the data is copied..
#@numba.guvectorize([(numba.float64[:],numba.float64, numba.float64[:])],'(n),()->(n)')
#def meanx1(data: [float], blank: float , meanx: [float]):
 #  meanx = data / 1.0
  # blank = blank
   #meanx[:-1] += data [1:]
   #meanx[1:] += data [:-1]
   #meanx[1:-1] /= 3
   #meanx[0] /= 2
   #meanx[-1] /= 2
 # return meanx #simple, easy, 1-dimensional function

@numba.jit#((numba.float64[::])(numba.float64[::]))
def meanx2(data:[float]):
    mean = data / 1.0
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

#TODO This function takes 1600 milliseconds on the first pass. Why?
@numba.jit
def posterior_mean_gen(prior_mean: [float],prior_variance: [float],data: [float],data_variance: [float],posterior_variance: [float]):
    return ( prior_mean / prior_variance + data / data_variance ) * posterior_variance


#notes: this particular function call takes on average 70ms to 100ms to complete. It cannot be optimized.
def chi2_pdf_call(data: [float],size):
    #start = timer()
    pdf = scipy.stats.chi2.pdf(data, df=size)
    return pdf

@numba.jit((numba.float64[:])( numba.float64[:]))
def nan(data: [float]):
    return numpy.asarray([x if not math.isnan(x) else 0.0 for x in data])

@numba.jit(numba.int16( numba.int16,numba.int16,numba.int16))
def bound(value: int, low: int = 20, high: int =100):
     diff = high - low
     return (((value - low) % diff) + low)

@numba.jit((numba.float64[:])( numba.float64[:]))
def variance(data: [float]):

    #note: nothing in this function is set in stone or even good.
    #the contents of this function determine how fabada sees the need for convolution.
    #i guess. i dont really know. But this is the source of the clicking, the dropped samples, etc.
    #the only time fabada will work for this particular application is when we get this right.

    data1 = data / 1.0 #get a copy of data
    data_alpha_padded = numpy.concatenate(
        (numpy.full((1,), (data1[0] / 2) + (data1[1] / 2)), data1, numpy.full((1,), (data1[-1] / 2) + (data1[-2] / 2))))
    data_beta = numpy.asarray([(i + j + k / 3) for i, j, k in
                               zip(data_alpha_padded, data_alpha_padded[1:], data_alpha_padded[2:])])
    #a bit like a weiner filter, this is a averaging filter that helps find the smooth mean
    data_mean = numpy.mean(data_beta) #get the mean
    #The formula for standard deviation is the square root of the sum of squared differences from the mean divided by the size of the data set.
    data_variance= numpy.asarray([(abs(data_mean - x)) ** 2 for x in data1])
    #print (data_variance)
    #print(numpy.mean(data_variance),(numpy.mean(data1)))
    #the algorithm OCCASIONALLY takes a giant shit
    #but WHY is it taking a giant shit??
    #[74329902.55352211 67757248.82468097 51918929.63034503...<bad
     #60349268.14146699 52337688.42043835 58483937.18969833]
   # 157455248.3786176(9012.660807291666, 9012.660807291666)<the consequence of bad
    #[74296660.81459373 67758433.17678991 51905556.25920093...<bad
    # 60365923.97919876 52353199.42613887 58500333.61792489]<bad
   # 157456076.8060095(9012.691569010416, 9012.691569010416)<bad
   # [1.02672646e+08 1.18652028e+08 1.16203035e+08... 2.19091838e+08 < good
   #  2.12510829e+08 2.16170892e+08]
   # 197914570.7785603(10280.705891927084, 10280.705891927084)<good
    #data_variance= nan(data_variance)

    # bayes = numpy.asarray([j if x!=0 else x for j,x in zip(bayes,data)])

    # data_variance = numpy.asarray([j if x<1 else x for j,x in zip(data_variance,data)])
    return data_variance



class Filter(object):

  #  def __init__(self):
       # self._processing_size = processing_size
      #  self._sample_rate = sample_rate
     #note: if removing staticmethod, insert self, before data
    @staticmethod
    def fabada(
          data: [float],
          data_variance: float = 1,
          max_iter: int = 200,
          verbose: bool = False,
          **kwargs
    ) -> numpy.array:

        """
        FABADA for any kind of data (1D or 2D). Performs noise reduction in input.
        :param data: Noisy measurements, either 1 dimension (M) or 2 dimensions (MxN)
        :param data_variance: Estimated variance of the input, either MxN array, list
                            or float assuming all point have same variance.
        :param max_iter: 3000 (default). Maximum of iterations to converge in solution.
        :param verbose: False (default) or True. Spits some informations about process.
        :param **kwargs: Future Work.
        :return bayes: denoised estimation of the data with same size as input.
         """
        #data = numpy.array(data / 1.0)
        data = data.astype(dtype=numpy.float64) # 9 milliseconds less than numpy.array, only works if passing a numpy array already
        #data = numpy.asarray([x if not math.isnan(x) else 0.0 for x in data])
        #nan_to_num for this dataset is 90 milliseconds. numpy.asarray([x if not math.isnan(x) else 0.0 for x in data]) is 4,226!
        #however, numba accelerated, it's still faster!

        if verbose:
            if data.ndim == 1:
                print("FABADA 1-D initialize")
            elif data.ndim == 2:
                print("FABADA 2-D initialize")
            else:
                print("Warning: Size of array not supported")
            t=timer()
            numpy.seterr(all='raise')


        data = nan(data) #sanitize your data
        min = numpy.finfo(numpy.float64).eps

        if (data.ndim < 1 or data.ndim >2 ):
            print("number of dimensions not supported~!")
            return #remove this logic from a loop that runs 100 times

         #data_variance = numpy.array(data_variance / 1.0)

        if not kwargs:
            kwargs = {}
            kwargs["debug"] = False

        data_variance = variance(data)
        data_variance[data_variance == 0] = min
        #initialize bayes for the function return
        bayes = 0.0 * numpy.ones_like(data)

        try:
        # INITIALIZING ALGORITMH ITERATION ZERO

            posterior_mean = data
            posterior_variance = data_variance


            evidence = Evidence1st(0.0, numpy.sqrt(data_variance), 0.0, data_variance)
            initial_evidence = evidence
            chi2_pdf, chi2_data, iteration = 0, data.size, 0
            chi2_pdf_derivative, chi2_data_min = 0, data.size
            bayesian_weight = 0.0
            bayesian_model = 0.0
            #converged = False
            iteration += 1  # set  number of iterations done

            chi2_pdf_previous = chi2_pdf
            chi2_pdf_derivative_previous = chi2_pdf_derivative
            evidence_previous = numpy.mean(evidence)

            iteration += 1  # Check number of iterartions done

        # GENERATES PRIORS
            if (posterior_mean.ndim == 1):
                prior_mean = meanx1(posterior_mean)
            if (posterior_mean.ndim == 2):
                prior_mean = meanx2(posterior_mean)

          #if data dimensions are not 1 or 2, it will break further up in the loop.
            prior_variance = posterior_variance

        # APPLIY BAYES' THEOREM
        # prevent le' devide by le zeros

            posterior_variance = 1 / (1 / prior_variance + 1 / data_variance)
            posterior_mean = posterior_mean_gen(prior_mean, prior_variance, data, data_variance, posterior_variance)

        # EVALUATE EVIDENCE
            evidence = Evidence(prior_mean, data, prior_variance, data_variance)
            evidence_derivative = numpy.mean(evidence) - evidence_previous

        # EVALUATE CHI2
            chi2_data = numpy.sum((data - posterior_mean) ** 2 / data_variance)

            chi2_pdf = chi2_pdf_call(chi2_data, data.size)
            chi2_pdf_derivative = chi2_pdf - chi2_pdf_previous
            chi2_pdf_snd_derivative = chi2_pdf_derivative - chi2_pdf_derivative_previous

         # COMBINE MODELS FOR THE ESTIMATION

            model_weight = evidence * chi2_data
            bayesian_weight += model_weight
            bayesian_model += model_weight * posterior_mean
            chi2_data_min = chi2_data

            while 1:

                if (
                    (chi2_data > data.size and chi2_pdf_snd_derivative >= 0)
                    and (evidence_derivative < 0)
                    or (iteration > max_iter)
                    ):
                    #converged = True
                    break #break the loop when we're done by prematurely setting converged and returning
                    #this allows us to test if one round of convergence is enough.
                #else:# the else here is redundant, as we either do or dont do, continue resets the while
                #and break terminates. While 1 is the fastest we can operate in this mode

                chi2_pdf_previous = chi2_pdf
                chi2_pdf_derivative_previous = chi2_pdf_derivative
                evidence_previous = numpy.mean(evidence)

                iteration += 1  # Check number of iterartions done

                    # GENERATES PRIORS
                if(posterior_mean.ndim ==1):
                     prior_mean = meanx1(posterior_mean)
                if(posterior_mean.ndim == 2):
                     prior_mean = meanx2(posterior_mean)
                prior_variance = posterior_variance

                # APPLIY BAYES' THEOREM
                 #prevent le' devide by le zeros
                #prior_variance[prior_variance == 0] = min
                #data_variance[data_variance == 0] = min
                #edit: now that we run once, this code isnt needed?

                posterior_variance = 1 / (1 / prior_variance + 1 / data_variance)
                posterior_mean =  posterior_mean_gen(prior_mean,prior_variance,data,data_variance,posterior_variance)

                    # EVALUATE EVIDENCE
                evidence = Evidence(prior_mean, data, prior_variance, data_variance)
                evidence_derivative = numpy.mean(evidence) - evidence_previous

                     # EVALUATE CHI2
                chi2_data = numpy.sum((data - posterior_mean) ** 2 / data_variance)
                chi2_pdf = chi2_pdf_call(chi2_data, data.size)
                chi2_pdf_derivative = chi2_pdf - chi2_pdf_previous
                chi2_pdf_snd_derivative = chi2_pdf_derivative - chi2_pdf_derivative_previous

                    # COMBINE MODELS FOR THE ESTIMATION
                model_weight = evidence * chi2_data
                bayesian_weight += model_weight
                bayesian_model += model_weight * posterior_mean

                  # COMBINE ITERATION ZERO
            model_weight = initial_evidence * chi2_data_min
            bayesian_weight += model_weight
            bayesian_model += model_weight * data


            bayes = bayesian_model / bayesian_weight



            bayes =  nan(bayes)
            # don't accidentally insert garbage into our stream

        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

        #from now on using pyutils/line profiler to profile the code

        if verbose:
            print(
                "Finish at {} iterations".format(iteration),
                " and with an execute time of : ", int(1000* (timer() - t)), "ms"
            )

        return bayes #this will either return zeros or the desired data


class FilterRun(Thread):
    def __init__(self,rb,pb,channels,processing_size,dtype):
        super(FilterRun, self).__init__()
        self.running = True
        self.filter = Filter()
        self.rb = rb
        self.processedrb = pb
        self.channels = channels
        self.processing_size = processing_size
        self.dtype = dtype
        self.buffer = numpy.ndarray(dtype=self.dtype, shape=[int(self.processing_size * self.channels)])
        self.buffer = self.buffer.reshape(-1,self.channels)
    def write_filtered_data(self):
        audio = self.rb.read(self.processing_size)
        for i in range(self.channels):
            filtered = self.filter.fabada(audio[:, i])
            self.buffer[:, i] = filtered
        self.processedrb.write(self.buffer,error=False)

    def run(self):
        while 1:
            if len(self.rb) < self.processing_size * 2:
                time.sleep(0.001)
            else:
                self.write_filtered_data()
                #time.sleep(0.01)

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

    # From mozilla:

    # ''Stereo audio is probably the most commonly used channel arrangement in web audio, and 16-bit samples are used for the majority of day-to-day audio in use today.
    # For 16-bit stereo audio, each sample taken from the analog signal is recorded as two 16-bit integers, one for the left channel and one for the right.
    # That means each sample requires 32 bits of memory.
    # At the common sample rate of 48 kHz (48,000 samples per second), this means each second of audio occupies 192 kB of memory. ''

    #processingsize determination
    #a single second of data requires 192kb in stereo, per mozilla. It requires 96kb in mono.
    #we sample twice as much data as our processingsize indicates, because that is per-channel.
    #1 sample frames = 16 bits. but remember, our buffer is pulling twice that and running it twice.
    # so whatever processing size we have, is accurate against 96kb. 98304 bits(96kb) -> 6144 frames= 1 second.
    # the clicks are still visible at around 100ms intervals when sampling this way.
    #it's not clear why.
    #depending on the sample size, the clicks are more or less pronounced.
    #Additionally, the clicks have to do with the variance calculation. If I could improve this...
    #fabada does not do well with small windows.
    #6.144 frames per ms. About 2150 for a 350ms format window
    #with this setting, fabada's output is NOT less noisy!
    #By setting it to 6144 we have a good interval to compute noise over. 1 second delay.



    def __init__(self, processing_size=6144, sample_rate=48000, channels=2, buffer_delay=1.5, # or 1.5, measured in seconds
                 micindex=1, speakerindex=1, dtype=numpy.float32):
        self.pa = pyaudio.PyAudio()
        self._processing_size = processing_size
        self.filter = Filter()

        # np_rw_buffer (AudioFramingBuffer offers a delay time)
        self._sample_rate = sample_rate
        self._channels = channels
        # self.rb = RingBuffer((int(sample_rate) * 5, channels), dtype=numpy.dtype(dtype))
        self.rb = AudioFramingBuffer(sample_rate=sample_rate, channels=channels,
                                     seconds=5,  # Buffer size (need larger than processing size)[seconds * sample_rate]
                                     buffer_delay=0,  # #this buffer doesnt need to have a size
                                     dtype=numpy.dtype(dtype))

        self.processedrb = AudioFramingBuffer(sample_rate=sample_rate, channels=channels,
                                     seconds=5,  # Buffer size (need larger than processing size)[seconds * sample_rate]
                                     buffer_delay=1.2, #give us a 100ms window
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
        self.filterthread.start()
        # Don't do this here. Do it in main. Other things may want to run while this stream is running
        # while self.micstream.is_active():
        #     eval(input("main thread is now paused"))

    listen = stream_start  # Just set a new variable to the same method


if __name__ == "__main__":
    SS = StreamSampler(buffer_delay=0)
    SS.listen()

    while SS.is_running():
        inp = input('Press enter to quit!\n')   # Halt until user input
        break

    SS.stop()
