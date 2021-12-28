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

"""
import numpy
import pyaudio
import scipy.stats
import numba
from np_rw_buffer import RingBuffer, AudioFramingBuffer
from timeit import default_timer as timer

@numba.jit #((numba.float64[:],numba.float64)(numba.float64[:], numba.float64))
def variance(data: [float]):
        data = data / 1.0
        #data_alpha_padded = numpy.empty(shape=[processing_size], dtype=float, order='C', like=None)
        #data_beta = numpy.empty(shape=[processing_size], dtype=float, order='C', like=None)
        #data_variance_residues = numpy.empty(shape=[processing_size], dtype=float, order='C', like=None)
        #data_variance  = numpy.empty(shape=[processing_size], dtype=float, order='C', like=None)
        #numpy.ndarray(buffer=data,dtype=float,shape=[16384])
        data_alpha_padded = numpy.concatenate((numpy.full((1,), (data[0] / 2) + (data[1] / 2)), data, numpy.full((1,), (data[-1] / 2) + (data[-2] / 2))))
        # average the data
        data_beta = numpy.asarray([(i + j + k / 3) for i, j, k in
                               zip(data_alpha_padded, data_alpha_padded[1:], data_alpha_padded[2:])])

        # get the smallest positive average, get the smallest out of the two. conveniently this also returns the distance between the average and the not so average
        #which is also known as the variance. ugh! numba why u like this
        #data_variance_residues = [abs(x - j) for x, j in zip(data_beta, data)]
        data_variance_residues = numpy.absolute(data_beta - data)
        # we assume beta is larger than residual. after all, few values will be outliers.
        # we want the algorithm to speculatively assume the variance is smaller for data that slopes well per sample.
        variance5 = abs(numpy.var(data_variance_residues)) * 1.61803398875

        data_variance =  data_variance_residues * variance5
        #for some reason sometimes this overflows to NAN, which is a major NONO
        #however for numpa to work the Nan processing has to be done elsewhere, thus export variance5 for the bounding
        return data_variance,variance5

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

@numba.jit #((numba.float64[:],numba.float64[:],numba.float64,numba.float64)(numba.float64))
def evaluate(prior_mean: [float],data: [float],prior_variance: float,data_variance: float):
        x = numpy.exp(-((prior_mean - data) ** 2) / (2 * (prior_variance + data_variance))) / numpy.sqrt(
            2 * numpy.pi * (prior_variance + data_variance))
        return x

@numba.jit #((numba.float64[:])(numba.float64))
def Evidence_start(data_variance: [float]):
        x = numpy.exp(-((0 - numpy.sqrt(data_variance)) ** 2) / (2 * (0 + data_variance))) / numpy.sqrt(
            2 * numpy.pi * (0 + data_variance))
        return x

@numba.jit ((numba.float64[:])(numba.float64[:]))
def running_mean(data: [float]):
    meany = data / 1.0
    meanx = data / 1.0
    meanx[:-1] += meany[1:]
    meanx[1:] += meany[:-1]
    meanx[1:-1] /= 3
    meanx[0] /= 2
    meanx[-1] /= 2
    return meanx





#notes: this particular function call takes on average 70ms to 100ms to complete
def chi2_pdf_call(data: [float],size):
    #start = timer()
    pdf = scipy.stats.chi2.pdf(data, df=size)
    #end = timer()
    #time = (end - start) * 1000000.00  # Time in seconds, e.g. 5.38091952400282
    #print("chi2.pdf took : ", time, " ms!")
    return pdf

class Filter(object):

  #  def __init__(self):
       # self._processing_size = processing_size
      #  self._sample_rate = sample_rate

    def __PSNR(self,recover, signal, L=255):
        MSE = numpy.sum((recover - signal) ** 2) / (recover.size)
        return 10 * numpy.log10((L) ** 2 / MSE)

    def fabada(self,data: [float]):
        # fabada expects the data as a floating point array, so, that is what we are going to work with.
        #define the number of iterations based on the SNR of the sample, where noisier samples need more work.
        # the most this can be is around 40 to -80, inverted, adds max of 80, subtracts most of 40.
        #If the SNR is high, fabada doesnt seem to be able to do as much. Perhaps I am wrong.
        # Get the channels
        data = data.astype(float)

        #todo: perhaps do better SNR/variance calculations based on power estimation?
        #P_s = 1;   % target signal power
        #SNR = 15;  % target SNR in dB
        #P_n = P_s / 10^(SNR/10); % calculated noise power
        #s = randn(1,N)*sqrt(P_s);
        #v = randn(1,N)*sqrt(P_n);
        #y = a*s + v;
        #-50 at most we want to do is 120 cycles. The least we want to do is ~70
        #the problem, at the moment, is that this calculation is being done regardless of passband width
        #which means the narrower the passband the higher the SNR is- fewer states, higher mean
        #also, this results in MORE noise being injected into the stream, because FABAS doesn't know about
        #passbands and so doesnt know what it's being fed has a high-pass filter on it.
        #as a result, number of cycles has to be hardcoded
        max_iter: int = 100 #+ int(numpy.nan_to_num(signaltonoise_dB(data),neginf=-50,posinf=50)* -1.0)
        # move buffer calculations
        posterior_mean = data
        data_variance ,var5 = variance(data)
        # prevents overflows
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

            prior_mean = running_mean(posterior_mean)
            prior_variance = posterior_variance

        # APPLIY BAYES' THEOREM
            posterior_variance = 1 / (1 / prior_variance + 1 / data_variance)
            posterior_mean = (prior_mean / prior_variance + data / data_variance) * posterior_variance

            # EVALUATE EVIDENCE
            evidence = evaluate(prior_mean,data,prior_variance,data_variance)
            evidence_derivative = numpy.mean(evidence) - evidence_previous

            # EVALUATE CHI2
            chi2_data = numpy.sum((data - posterior_mean) ** 2 / data_variance)


            chi2_pdf =  chi2_pdf_call(chi2_data, data.size)

            start = timer()
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

        data = bayesian_model /bayesian_weight
        return data

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

    def __init__(self, processing_size=16384, sample_rate=48000, channels=2, buffer_delay=1.5, # or 1.5, measured in seconds
                 micindex=1, speakerindex=1, dtype=numpy.int32): #int32 works just as fast as int16, maybe adds some precision
        self.pa = pyaudio.PyAudio()
        self._processing_size = processing_size
        self.filter = Filter()

        # np_rw_buffer (AudioFramingBuffer offers a delay time)
        self._sample_rate = sample_rate
        self._channels = channels
        # self.rb = RingBuffer((int(sample_rate) * 5, channels), dtype=numpy.dtype(dtype))
        self.rb = AudioFramingBuffer(sample_rate=sample_rate, channels=channels,
                                     seconds=5,  # Buffer size (need larger than processing size)[seconds * sample_rate]
                                     buffer_delay=buffer_delay,  # Save data for 1 second then start playing
                                     dtype=numpy.dtype(dtype))

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
        except AttributeError:
            pass
        try:  # AudioFramingBuffer
            self.rb.sample_rate = value
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
        except AttributeError:
            pass
        try:  # AudioFrammingBuffer
            self.rb.channels = value
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
                              frames_per_buffer= int(self._processing_size),  # Don't really need this? Default is 1024 I think
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
    # memoryview avoids the copy by merely being a view
    def non_blocking_stream_read(self, in_data, frame_count, time_info, status):
        audio_in = memoryview(numpy.ndarray(buffer=memoryview(in_data), dtype=self.dtype, shape=[int(self._processing_size* self._channels)]).reshape(-1, self.channels))
        self.rb.write(audio_in, error=False)
        return None, pyaudio.paContinue

    def non_blocking_stream_write(self, in_data, frame_count, time_info, status):
        # Read raw data
        # filtered = self.rb.read(frame_count)
        # if len(filtered) < frame_count:
        #     filtered = numpy.zeros((frame_count, self.channels), dtype=self.dtype)

        filtered = self.process_audio()
        byts_out = filtered.astype(self.dtype).tobytes()
        return byts_out, pyaudio.paContinue

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

    def process_audio(self):
        if len(self.rb) < self.processing_size:
            print('Not enough data in the buffer! Increase the buffer_delay')
            return numpy.zeros((self.processing_size, self.channels), dtype=self.dtype)

        audio = self.rb.read(self.processing_size)
        chans = []
        for i in range(self.channels):
            filtered = self.filter.fabada(audio[:, i])
            chans.append(filtered)

        # Could buffer again then the output would just take from the buffer
        return numpy.column_stack(chans)


if __name__ == "__main__":
    SS = StreamSampler(buffer_delay=0)
    SS.listen()

    while SS.is_running():
        inp = input('Press enter to quit!\n')   # Halt until user input
        break

    SS.stop()
