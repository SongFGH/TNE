from gensim.models.base_any2vec import *


class TNEBaseAny2VecModel(BaseAny2VecModel):

    def train_topic(self, number_of_topics, data_iterable, epochs=None, total_examples=None,
              total_words=None, queue_factor=2, report_delay=1.0, callbacks=(), **kwargs):
        """Handle multi-worker training."""
        self._set_train_params(**kwargs)
        if callbacks:
            self.callbacks = callbacks
        self.epochs = epochs
        self._check_training_sanity(
            epochs=epochs,
            total_examples=total_examples,
            total_words=total_words, **kwargs)

        for callback in self.callbacks:
            callback.on_train_begin(self)

        trained_word_count = 0
        raw_word_count = 0
        start = default_timer() - 0.00001
        job_tally = 0

        for cur_epoch in range(self.epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(self)

            trained_word_count_epoch, raw_word_count_epoch, job_tally_epoch = self._train_epoch_topic(
                data_iterable, cur_epoch=cur_epoch, total_examples=total_examples, total_words=total_words,
                queue_factor=queue_factor, report_delay=report_delay)
            trained_word_count += trained_word_count_epoch
            raw_word_count += raw_word_count_epoch
            job_tally += job_tally_epoch

            for callback in self.callbacks:
                callback.on_epoch_end(self)

        # Log overall time
        total_elapsed = default_timer() - start
        self._log_train_end(raw_word_count, trained_word_count, total_elapsed, job_tally)

        self.train_count += 1  # number of times train() has been called
        self._clear_post_train()

        for callback in self.callbacks:
            callback.on_train_end(self)
        return trained_word_count, raw_word_count

    def _train_epoch_topic(self, data_iterable, cur_epoch=0, total_examples=None,
                           total_words=None, queue_factor=2, report_delay=1.0):
        """Train one epoch."""
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [
            threading.Thread(
                target=self._worker_loop_topic,
                args=(job_queue, progress_queue,))
            for _ in xrange(self.workers)
        ]

        workers.append(threading.Thread(
            target=self._job_producer,
            args=(data_iterable, job_queue),
            kwargs={'cur_epoch': cur_epoch, 'total_examples': total_examples, 'total_words': total_words}))

        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        trained_word_count, raw_word_count, job_tally = self._log_epoch_progress(
            progress_queue, job_queue, cur_epoch=cur_epoch, total_examples=total_examples, total_words=total_words,
            report_delay=report_delay)

        return trained_word_count, raw_word_count, job_tally

    def _worker_loop_topic(self, job_queue, progress_queue):
        """Train the model, lifting lists of data from the job_queue."""
        thread_private_mem = self._get_thread_working_mem()
        jobs_processed = 0
        while True:
            job = job_queue.get()
            if job is None:
                progress_queue.put(None)
                break  # no more jobs => quit this worker
            data_iterable, job_parameters = job

            for callback in self.callbacks:
                callback.on_batch_begin(self)

            tally, raw_tally = self._do_train_job_topic(data_iterable, job_parameters, thread_private_mem)

            for callback in self.callbacks:
                callback.on_batch_end(self)

            progress_queue.put((len(data_iterable), tally, raw_tally))  # report back progress
            jobs_processed += 1
        logger.debug("worker exiting, processed %i jobs", jobs_processed)

    def _do_train_job_topic(self, data_iterable, job_parameters, thread_private_mem):
        """Train a single batch. Return 2-tuple `(effective word count, total word count)`."""
        raise NotImplementedError()


class TNEBaseWordEmbeddingsModel(TNEBaseAny2VecModel, BaseWordEmbeddingsModel):

    def train_topic(self, number_of_topics, sentences, total_examples=None, total_words=None,
              epochs=None, start_alpha=None, end_alpha=None, word_count=0,
              queue_factor=2, report_delay=1.0, compute_loss=False, callbacks=()):

        self.alpha = start_alpha or self.alpha
        self.min_alpha = end_alpha or self.min_alpha
        self.compute_loss = compute_loss
        self.running_training_loss = 0.0
        return super(TNEBaseWordEmbeddingsModel, self).train_topic(number_of_topics,
            sentences, total_examples=total_examples, total_words=total_words,
            epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha, word_count=word_count,
            queue_factor=queue_factor, report_delay=report_delay, compute_loss=compute_loss, callbacks=callbacks)