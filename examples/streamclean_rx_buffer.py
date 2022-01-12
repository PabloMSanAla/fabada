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
The program expects 44100hz audio, 16 bit, two channel, but can be configured to work with anything thanks to Justin Engel.

"""
import numpy
import pyaudio
import numba
from np_rw_buffer import AudioFramingBuffer
from threading import Thread
import math
import time

@numba.jit(numba.float64[:](numba.float64[:], numba.float64, numba.float64),nopython=True,parallel=True,nogil=True,cache=True)
def numba_fabada(data: [numpy.float64], timex: numpy.float64, work: numpy.float64):
        #notes:
        #The pythonic way to COPY an array is to do x[:] = y[:]
        #do x=y and it wont copy it, so any changes made to X will also be made to Y.
        #also, += does an append instead of a +
        #math.sqrt is 7x faster than numpy.sqrt but not designed for complex numbers.
        #specifying the type and size in advance of all variables accelerates their use, particularily when JIT is used.
        #However, JIT does not permit arbitrary types like UNION? maybe it does but i havnt figured out how.
        #this implementation uses a lot of for loops because they can be easily vectorized by simply replacing
        #range with numba.prange and also because they will translate well to other languages
        #this implementation of FABADA is not optimized for 2d arrays, however, it is easily swapped by changing the means
        #estimation and by simply changing all other code to iterate over 2d instead of 1d
        #care must be taken with numba parallelization/vectorization

        with numba.objmode(start=numba.float64):
            start = time.time()

        iterations: int = 1
        TAU: numpy.float64 = 2 * math.pi
        N = data.size

        #must establish zeros for the model or otherwise when data is empty, algorithm will return noise
        bayesian_weight  = numpy.zeros_like(data)
        bayesian_model = numpy.zeros_like(data)
        model_weight = numpy.zeros_like(data)

        #pre-declaring all arrays allows their memory to be allocated in advance
        posterior_mean  = numpy.empty_like(data)
        posterior_variance  = numpy.empty_like(data)
        initial_evidence = numpy.empty_like(data)
        evidence = numpy.empty_like(data)
        prior_mean = numpy.empty_like(data)
        prior_variance = numpy.empty_like(data)

        #working set arrays, no real meaning, just to have work space
        ja1 = numpy.empty_like(data)
        ja2 = numpy.empty_like(data)
        ja3 = numpy.empty_like(data)
        ja4 = numpy.empty_like(data)

        #eliminate divide by zero
        data[data == 0.0] = 2.22044604925e-16
        min_d: numpy.float64 = numpy.min(data)
        max_d: numpy.float64 = numpy.ptp(data)
        min: numpy.float64 = 2.22044604925e-16
        max:numpy.float64 =  44100
        #the higher max is, the less "crackle".
        #The lower floor is, the less noise reduction is possible.
        #floor can never be less than max/2.
        #noise components are generally higher frequency.
        #the higher the floor is set, the more attenuation of both noise and signal.


        data: [numpy.float64] =  numpy.interp(data, (min_d, max_d),(min, max))#normalize the data

        posterior_mean[:] = data[:]
        prior_mean[:] = data[:]
        data_mean = numpy.mean(data)  # get the mean
        data_variance = numpy.empty_like(data)

        for i in numba.prange(N):
            data_variance[i] = numpy.abs(data_mean - (data[i]+max)) ** 2

        posterior_variance[:] = data_variance[:]

        for i in numba.prange(N):
            ja1[i] = ((0.0 - math.sqrt(data[i])) ** 2)
            ja2[i] = ((0.0 + data_variance[i]) * 2)
            ja3[i] = math.sqrt(TAU * (0.0 + data_variance[i]))
        for i in numba.prange(N):
            ja4[i] = math.exp(-ja1[i] / ja2[i])
        for i in numba.prange(N):
            evidence[i] = ja4[i] / ja3[i]
        evidence_previous: numpy.float64 = numpy.mean(evidence)
        initial_evidence[:] = evidence[:]

        for i in numba.prange(N):
            ja1[i] = data[i] - posterior_mean[i]
        for i in numba.prange(N):
            ja1[i] = ja1[i] ** 2.0 / data_variance[i]
        chi2_data_min: numpy.float64 = numpy.sum(ja1)
        chi2_pdf_previous: numpy.float64 = 0.0
        chi2_pdf_derivative_previous: numpy.float64 = 0.0
        # COMBINE MODELS FOR THE ESTIMATION


        while 1:

        # GENERATES PRIORS
            prior_mean[:] = posterior_mean[:]
            prior_mean[:-1] += posterior_mean[1:]
            prior_mean[1:] += posterior_mean[:-1]
            prior_mean[1:-1] /= 3
            prior_mean[0] /= 2
            prior_mean[-1] /= 2

            # APPLY BAYES' THEOREM ((b\a)a)\b?
            prior_variance[:] = posterior_variance[:]

            for i in numba.prange(N):
                posterior_variance[i] = 1.0 / (1.0/ data_variance[i] + 1.0 / prior_variance[i])
            for i in numba.prange(N):
                posterior_mean[i] = (((prior_mean[i] / prior_variance[i]) + ( data[i] / data_variance[i])) * posterior_variance[i])

            # EVALUATE EVIDENCE

            for i in numba.prange(N):
                ja1[i] = ((prior_mean[i] - math.sqrt(data[i])) ** 2)
                ja2[i] = ((prior_variance[i] + data_variance[i]) * 2)
                ja3[i] = math.sqrt(TAU * (prior_variance[i] + data_variance[i]))
            for i in numba.prange(N):
                ja4[i] = math.exp(-ja1[i] / ja2[i])
            for i in numba.prange(N):
                evidence[i] = ja4[i] / ja3[i]

            evidence_derivative: numpy.float64 = numpy.mean(evidence) - evidence_previous

            # EVALUATE CHI2

            for i in numba.prange(N):
                ja1[i] = ((data[i] - posterior_mean[i]) ** 2 / data_variance[i])
            chi2_data = numpy.sum(ja1)


            # COMBINE MODELS FOR THE ESTIMATION
            for i in numba.prange(N):
                model_weight[i] = evidence[i] * chi2_data

            for i in numba.prange(N):
                bayesian_weight[i] = bayesian_weight[i] + model_weight[i]
                bayesian_model[i] = bayesian_model[i] + (model_weight[i] * posterior_mean[i])

            df: int = 5  # note: this isnt the right way to use this function. DF is supposed to be, like data.size but that would be enormous
            #for any data set which is non-trivial. Remember, DF/2 - 1 becomes the exponent! For anything over a few hundred this quickly exceeds float64.
            ## chi2.pdf(x, df) = 1 / (2*gamma(df/2)) * (x/2)**(df/2-1) * exp(-x/2)
            gammar: numpy.float64 = (2. * math.lgamma(df / 2.))
            gammaz: numpy.float64 = ((df / 2.) - 1.)
            gamman: numpy.float64 = (chi2_data / 2.)
            gammas: numpy.float64 = (numpy.sign(gamman) * (
                        (abs(gamman)) ** gammaz))  #TODO
            if math.isnan(gammas):
                gammas = (numpy.sign(gamman) * ((abs(gamman)) * gammaz))
            gammaq: numpy.float64 = math.exp(-chi2_data / 2.)
            #for particularily large values, math.exp just returns 0.0
            #TODO
            gammaa: numpy.float64 = 1. / gammar
            chi2_pdf = gammaa * gammas * gammaq

            # COMBINE MODELS FOR THE ESTIMATION

            chi2_pdf_derivative = chi2_pdf - chi2_pdf_previous
            chi2_pdf_snd_derivative = chi2_pdf_derivative - chi2_pdf_derivative_previous
            chi2_pdf_previous = chi2_pdf
            chi2_pdf_derivative_previous = chi2_pdf_derivative
            evidence_previous: numpy.float64 = evidence_derivative

            with numba.objmode(current=numba.float64):
                current = time.time()
            timerun = (current - start) * 1000

            if (
                    (int(chi2_data) > data.size and chi2_pdf_snd_derivative >= 0)
                    or (abs(evidence_derivative) < 0)
                    or (timerun > int(timex))  # use no more than 95% of the time allocated per cycle
            ):
                break

            iterations += 1


            # COMBINE ITERATION ZERO
        for i in numba.prange(N):
            model_weight[i] = initial_evidence[i] * chi2_data_min
        for i in numba.prange(N):
            bayesian_weight[i] = (bayesian_weight[i]  + model_weight[i])
            bayesian_model[i] = bayesian_model[i] + ( model_weight[i] *  data[i])

        for i in numba.prange(N):
            data[i] = bayesian_model[i] / bayesian_weight[i]
        data = numpy.interp(data,(min, max), (min_d, +max_d))#denormalize the data
        print(iterations,timerun)
        return data
        

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
        self.buffer2 = numpy.ndarray(dtype=numpy.float64, shape=[int(self.processing_size * self.channels)])
        self.buffer2 = self.buffer.reshape(-1,self.channels)
        self.buffer = self.buffer.reshape(-1,self.channels)


    def write_filtered_data(self):
        #t = time.time()
        numpy.copyto(self.buffer,self.rb.read(self.processing_size).astype(dtype=numpy.float64))
        for i in range(self.channels):

                self.buffer2[:, i] = numba_fabada(self.buffer[:, i],495,44100)

        #numpy.copyto(self.buffer[:, i],numba_fabada(self.buffer[:, i]))
        self.processedrb.write(self.buffer2.astype(dtype=self.dtype),error=True)
        #x = time.time()
        #print((x - t)*1000)


    def run(self):
        while self.running:
            if len(self.rb) < self.processing_size * 2:
                time.sleep(0.4) #idk how long we should sleep
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
                 micindex=1, speakerindex=1, dtype=numpy.float32):
        self.pa = pyaudio.PyAudio()
        self._processing_size = sample_rate
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
                                     buffer_delay=0, # as long as fabada completes in O(n) of less than the sample size in time
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
    # after importing numpy, reset the CPU affinity of the parent process so
    # that it will use all cores
    SS = StreamSampler(buffer_delay=0)
    SS.listen()
    SS.filterthread.start()

    while SS.is_running():
        inp = input('Press enter to quit!\n')   # Halt until user input
        SS.filterthread.stop()
        SS.stop()
        break

    #SS.stop()
