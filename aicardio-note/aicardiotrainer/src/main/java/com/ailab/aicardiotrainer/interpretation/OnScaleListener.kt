package com.ailab.aicardiotrainer.interpretation

interface OnScaleListener {
    fun onScale(scaleFactor: Float, focusX: Float, focusY: Float)
}
