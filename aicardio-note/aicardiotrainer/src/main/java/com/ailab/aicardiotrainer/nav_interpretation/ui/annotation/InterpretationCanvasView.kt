/*
 * Copyright 2020 UET-AILAB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.ailab.aicardiotrainer.nav_interpretation.ui.annotation

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.*
import android.os.Bundle
import android.util.AttributeSet
import android.util.Log
import android.view.MotionEvent
import android.view.ScaleGestureDetector

import org.json.JSONException
import org.json.JSONObject
import java.util.logging.Logger

class InterpretationCanvasView : androidx.appcompat.widget.AppCompatImageView,
    OnScaleListener
{
    constructor(context: Context) : this(context, null)
    constructor(context: Context, attributes: AttributeSet?) : this(context, attributes, 0)
    constructor(context: Context, attributes: AttributeSet?, defStyleAttr : Int) : super(context, attributes, defStyleAttr)

    var isZooming: Boolean = false
    @Suppress("unused")
    private val log = Logger.getLogger(InterpretationCanvasView::class.java.name)
    private lateinit var pixelToValueMatrix : Matrix

    private var onNormalizeTouchListener: OnNormalizeTouchListener? = null
    private var onDrawListener : OnDrawListener? = null

    var bitmap : Bitmap? = null
    private var drawMatrix = Matrix()
    private var scale = 1.0F
    private var translateX = 0F
    private var translateY = 0F
    private var lastX = -1F
    private var lastY = -1F
    companion object {
        const val TAG = "InterpretationCanvasView"
        const val KEY_SCALE = "scale"
        const val KEY_TRANSLATE_X = "translateX"
        const val KEY_TRANSLATE_Y = "translateY"
        const val KEY_LAST_X = "lastX"
        const val KEY_LAST_Y = "lastY"
    }

    fun saveToBundle(savedInstanceState: Bundle) {
        savedInstanceState.putFloat(KEY_SCALE, scale)
        savedInstanceState.putFloat(KEY_TRANSLATE_X, translateX)
        savedInstanceState.putFloat(KEY_TRANSLATE_Y, translateY)
        savedInstanceState.putFloat(KEY_LAST_X, lastX)
        savedInstanceState.putFloat(KEY_LAST_Y, lastY)

    }

    fun getFromBundle(savedInstanceState: Bundle) {
        scale = savedInstanceState.getFloat(KEY_SCALE)
        translateX = savedInstanceState.getFloat(KEY_TRANSLATE_X)
        translateY = savedInstanceState.getFloat(KEY_TRANSLATE_Y)
        lastX = savedInstanceState.getFloat(KEY_LAST_X)
        lastY = savedInstanceState.getFloat(KEY_LAST_Y)
    }

    private val scaleDetector = ScaleGestureDetector(context,
        ScaleListener(this)
    )

    override fun onScale(scaleFactor: Float, focusX: Float, focusY: Float) {
        Log.w("InterpretationCanvasView", "start running onScale ${isZooming} ${bitmap?.width}")
        if (bitmap == null || !isZooming) return
        scale *= scaleFactor
        if (scale < 1F) scale = 1F
        if (scale > 10F) scale = 10F
//        log.warning("scale = $scale scaleFactor = $scaleFactor fx = $focusX, fy = $focusY")
        Log.w("InterpretationCanvasView", "after set scale")
        bitmap?.let { setCustomImageBitmap(it) }
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event: MotionEvent?): Boolean {
//        log.warning("calling scaleDetector")
        scaleDetector.onTouchEvent(event)

        event?.let {
            //            log.warning("zoom = $isZooming action = ${it.action} ${it.x} ${it.y}")
            if (isZooming) {
                when (it.action) {
                    MotionEvent.ACTION_DOWN -> {
                        lastX = it.x
                        lastY = it.y
                    }
                    MotionEvent.ACTION_UP -> {
                        lastX = -1F
                        lastY = -1F
                    }
                    MotionEvent.ACTION_MOVE -> {
                        if (lastX >= 0 && lastY >= 0) {
                            translateX += it.x-lastX
                            translateY += it.y-lastY
                            lastX = it.x
                            lastY = it.y
                        }
                        bitmap?.let { bm -> setCustomImageBitmap(bm) }
                    }
                    else -> {}
                }
            }
        }

        bitmap?.let {
            pixelToValueMatrix = Matrix()
            drawMatrix.invert(pixelToValueMatrix)
            val x = event?.x ?: -1F
            val y = event?.y ?: -1F
            val point = floatArrayOf(x, y)
            pixelToValueMatrix.mapPoints(point)
            val ix = point[0] / it.width
            val iy = point[1] / it.height
//            onNormalizeTouchListener?.onTouchEvent(this, event, ix, iy)
            if (ix in 0.0..1.0 && iy >= 0.0 && iy <= 1.0) {
                onNormalizeTouchListener?.onTouchEvent(this, event, ix, iy)
            }
        }
        return true
    }

    fun setOnNormalizeTouchListener(listener: OnNormalizeTouchListener) {
        onNormalizeTouchListener = listener
    }

    fun setOnDrawListener(listener: OnDrawListener) {
        onDrawListener = listener
    }

    // precondition: bitmap != null
    fun getScreenCoordinate(p: JSONObject) : FloatArray {
        return try {
            bitmap?.let {
                val x = p.getDouble("x") * it.width
                val y = p.getDouble("y") * it.height
                val point = floatArrayOf(x.toFloat(), y.toFloat())
                drawMatrix.mapPoints(point)
                point
            }?: run {
                floatArrayOf(0F, 0F)
            }

        } catch (e: JSONException) {
            Log.w(TAG, "getScreenCoordinate ${e}")
            floatArrayOf(0F, 0F)

        }

    }

    private class ScaleListener(val listener: OnScaleListener) : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        private val log = Logger.getLogger(ScaleListener::class.java.name)
        override fun onScale(detector: ScaleGestureDetector?): Boolean {
            return if (detector == null) false
            else {
                listener.onScale(detector.scaleFactor, detector.focusX, detector.focusY)
                true
            }
        }

        override fun onScaleBegin(detector: ScaleGestureDetector?): Boolean {
            detector?.let {
                log.warning("focusX = ${it.focusX}, focusY=${it.focusY}")
            }
            return true
        }
    }

    @SuppressLint("DrawAllocation")
    override fun onDraw(canvas: Canvas?) {
        super.onDraw(canvas)

        bitmap?.let { canvas?.drawBitmap(it, drawMatrix, Paint()) }

        val textPaint = Paint()
        textPaint.color = Color.RED
        textPaint.strokeWidth = 1.5F
        textPaint.textSize = 25.0F

        canvas?.drawText(infoText, 0F, 20F, textPaint)

        onDrawListener?.draw(this, canvas)

    }

    fun setFitScale(bmNull: Bitmap?) {
        bmNull?.let { bm ->
            val sX = width.toFloat() / bm.width.toFloat()
            val sY = height.toFloat() / bm.height.toFloat()
            translateX = 0F
            translateY = 0F
            scale = if (sX < sY) sX else sY
        }
    }

    fun setCustomImageBitmap(bmNull: Bitmap?) {
        bmNull?.let { bm ->
            bitmap = bm
            val left = (width - bm.width).toFloat() / 2 + translateX
            val top = (height - bm.height).toFloat() / 2 + translateY
//            Log.w(TAG, "setCustomImageBitmap width: ${width} height: ${height} img: w ${bm.width} h ${bm.height} -- x: ${translateX} ${translateY} --- left: ${left} top: ${top}")

            drawMatrix = Matrix()
            drawMatrix.setTranslate(left, top)
//            scale = 1.0F
            drawMatrix.preScale(scale, scale, bm.width.toFloat() / 2, bm.height.toFloat() / 2)
//            Log.w(TAG, "scale: ${scale} -- px: ${bm.width.toFloat() / 2} py: ${bm.height.toFloat() / 2}")

            invalidate()
        }
//        log.warning("setCustomImageBitmap width = ${bitmap?.width} height = ${bitmap?.height} scale = $scale left = $left top = $top width = $width height = $height")
    }

    var infoText : String = "0 / 0 / 0"


    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        bitmap?.let {
            setFitScale(it)
            setCustomImageBitmap(it)
        }
    }

    fun getScale() : Float {
        return scale
    }
}