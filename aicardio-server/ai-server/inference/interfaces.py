import time
import threading


__all__ = ["InferenceStep", "ThreadedStep"]


class InferenceStep(object):
    r"""Inference step, i.e. a single step in a inference pipeline. See
        inference.inferencer.Inferencer
    """
    def __init__(self, name, logger):
        r"""Initialize an inference step

        Args:
            name (str): The name of the step, e.g. FinetuneGeometricalFeatures
            logger (logging.Logger): The logger used to log the progress of the
                step, e.g.
                    logger = logging.getLogger("default")
        """
        self.name = name
        self.logger = logger
    
    def __call__(self, item):
        r"""Run an inference step on an item
        
        An item will be processed by the following steps
            1. Check whether all necessary inputs are contained in the item
            2. Run the main procedure and put the results into the item
            3. Check whether all desired outputs are contained in the item

        Args:
            item (dict): The item to be processed. This is a Python dict
                containing necessary inputs to the inference step
        Returns:
            dict: The input item with additional keys corresponding to the
                results of the inference step
        Example:
            A DICOMReader inference step should take an item of the form
                {
                    "dicom_path": <path to DICOM file>
                }
            and append a key "frames" to that item, i.e. the returned item is
                {
                    "dicom_path": <path to DICOM file>,
                    "frames": <frames read from DICOM file>
                }
        """
        self._validate_inputs(item)
        if item["is_valid"]:
            item = self._process(item)
        self._validate_outputs(item)
        return item
    
    def _validate_inputs(self, item):
        r"""Validate an input item

        This method should take an item, i.e. a Python dict, and add a key 
        "is_valid" (if not available) to the item. This key corresponds to a
        boolean value indicating whether the item is valid for running the step
        or not

        Args:
            item (dict): The item to be processed
        Example:
            def _validate_inputs(self, item):
                if "masks" in item:
                    item["valid"] = False
                else:
                    item["valid"] = True
        """
        raise NotImplementedError
    
    def _process(self, item):
        r"""Process an item

        Args:
            item (dict): The item to be processed
        Returns:
            dict: The input item with additional keys corresponding to the
                results of the inference step
        """
        raise NotImplementedError
    
    def _validate_outputs(self, item):
        r"""Validate an output item

        This method takes the item returned by self._process() and validate
        whether all desired outputs are contained in the item. This is similar
        to method self._validate_inputs()

        Args:
            item (dict): The item to be validated
        """
        raise NotImplementedError

    def visualize(self, item):
        r"""Visualize intermediate results or results of the inference step

        Args:
            item (dict): The item containing results to be visualized
        Returns:
            dict: The input item with additional key "visualize". This key
                correpsonds to a dict, whose keys are inference step's name, i.e.
                the name passed to the step constructor, and values are
                correpsonding list of cv2 images to be plotted or saved to disk
        """
        if item["is_valid"]:
            item = self._visualize(item)
        return item

    def _visualize(self, item):
        r"""Visualize intermediate results or results of the inference step
        
        This method serves for debugging purpose, i.e. visualize for debugging

        Args:
            item (dict): The item containing results to be visualized
        Returns:
            dict: The input item with additional key "visualize". This key
                corresponds to a list of cv2 images, which is then plotted or
                saved. See inference.inferencer.Inferencer.visualize()
        """
        raise NotImplementedError
    
    def log(self, item):
        r"""Log progress of an item

        Args:
            item (dict): The item to log
        """
        if item["is_valid"]:
            self._log(item)

    def _log(self, item):
        r"""Log progress of an item

        Args:
            item (dict): The item to log 
        """
        raise NotImplementedError


class ThreadedStep(threading.Thread):
    def __init__(self, operator, in_queue, out_queue):
        r"""Step class which supports multi-thread. 
        
        The thread will constantly get items from in_queue, process them, then
        put the resulting items into out_queue

        Args:
            operator (inference.interfaces.InferenceStep): Operator performed 
                by the thread
            in_queue (queue.Queue): Input queue
            out_queue (queue.Queue): Output queue
        """
        super().__init__()
        self.operator = operator
        self.in_queue = in_queue
        self.out_queue = out_queue

    def run(self):
        running = True
        while True:
            if not self.in_queue.empty():
                try:
                    item = self.in_queue.get()
                    item = self.operator(item)
                    item = self.operator.visualize(item)
                    self.out_queue.put(item)
                except Exception as e:
                    print(f"{e}")
            time.sleep(1e-6)
