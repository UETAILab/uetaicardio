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

package com.ailab.aicardiotrainer.annotationscreen

import android.app.Dialog
import android.content.Context
import android.graphics.*
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.view.MotionEvent
import android.view.Window
import android.widget.CompoundButton
import android.widget.Toast
import com.ailab.aicardiotrainer.R
import com.ailab.aicardiotrainer.interfaces.OnDrawListener
import com.ailab.aicardiotrainer.interfaces.OnNormalizeTouchListener
import com.ailab.aicardiotrainer.repositories.DicomDiagnosis
import kotlinx.android.synthetic.main.dialog_diagnosis.*
import org.json.JSONArray

class DiagnosisDialog(
    context: Context,
    private val listener: OnDiagnosisEnteredListener,
    dicomDiagnosis: DicomDiagnosis
) : Dialog(context), CompoundButton.OnCheckedChangeListener,
    OnNormalizeTouchListener,
    OnDrawListener {

    companion object {
        val TAG = "DiagnosisDialog"
        var bitmaps = HashMap<Int, Bitmap>()
        var bitmapDefault : Bitmap? = null
    }

    val userDiagnosis = DicomDiagnosis(dicomDiagnosis.toString())

    interface OnDiagnosisEnteredListener {
        fun onDiagnosisEntered(dicomDiagnosis: DicomDiagnosis)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        setContentView(R.layout.dialog_diagnosis)

        bt_cancel.setOnClickListener {
            this.cancel()
        }

        bt_OK.setOnClickListener {
            onOKClicked()
        }

        prepareImageViews(iv_diagnosis_2C)
        prepareImageViews(iv_diagnosis_3C)
        prepareImageViews(iv_diagnosis_4C)
        prepareImageViews(iv_diagnosis_pts_l)
        prepareImageViews(iv_diagnosis_pts_s)
        prepareImageViews(iv_diagnosis_no_label)


        cb_LAD.isChecked = userDiagnosis.lad
        cb_LCx.isChecked = userDiagnosis.lcx
        cb_RCA.isChecked = userDiagnosis.rca
        cb_is_standard_size.isChecked = userDiagnosis.isNotStandardImage

        cb_RCA.setOnCheckedChangeListener(this)
        cb_LCx.setOnCheckedChangeListener(this)
        cb_LAD.setOnCheckedChangeListener(this)
        cb_is_standard_size.setOnCheckedChangeListener(this)

        et_add_note.setText(userDiagnosis.note)
        et_add_note.addTextChangedListener(object : TextWatcher {
            override fun afterTextChanged(p0: Editable?) {
            }

            override fun beforeTextChanged(p0: CharSequence?, p1: Int, p2: Int, p3: Int) {
            }

            override fun onTextChanged(s: CharSequence?, p1: Int, p2: Int, p3: Int) {
                s?.let { userDiagnosis.note = it.toString() }
            }

        })
    }

    private fun onOKClicked() {
        listener.onDiagnosisEntered(userDiagnosis)
        dismiss()
    }

    private fun prepareImageViews(view: DrawCanvasView) {
        if (bitmapDefault == null) bitmapDefault = BitmapFactory.decodeResource(context.resources, R.drawable.heart)
        if (bitmaps.size == 0) {
            listOf(
                R.id.iv_diagnosis_2C,
                R.id.iv_diagnosis_3C,
                R.id.iv_diagnosis_4C,
                R.id.iv_diagnosis_pts_l,
                R.id.iv_diagnosis_pts_s,
                R.id.iv_diagnosis_no_label
            ).forEach {
                bitmaps.put(
                    it, BitmapFactory.decodeResource(
                        context.resources,
                        when (it) {
                            R.id.iv_diagnosis_2C -> R.drawable.heart_2c
                            R.id.iv_diagnosis_3C -> R.drawable.heart_3c
                            R.id.iv_diagnosis_4C -> R.drawable.heart_4c
                            R.id.iv_diagnosis_pts_l -> R.drawable.heart_pts_l
                            R.id.iv_diagnosis_pts_s -> R.drawable.heart_pts_s
                            R.id.iv_diagnosis_no_label -> R.drawable.heart_no_label
                            else -> R.drawable.heart_2c
                        }
                    )
                )
            }
        }
        setDiagramBackground()
        view.setCustomImageBitmap(bitmaps.getOrDefault(view.id, bitmapDefault))

        view.isZooming = false
        view.infoText = ""
        view.setOnNormalizeTouchListener(this)
        view.setOnDrawListener(this)
    }

    override fun onCheckedChanged(cb: CompoundButton?, p1: Boolean) {
        userDiagnosis.lad = cb_LAD.isChecked
        userDiagnosis.lcx = cb_LCx.isChecked
        userDiagnosis.rca = cb_RCA.isChecked
        userDiagnosis.isNotStandardImage = cb_is_standard_size.isChecked
    }

    override fun onTouchEvent(view: DrawCanvasView, event: MotionEvent?, ix: Float, iy: Float) {
        if (event == null || event.action != MotionEvent.ACTION_UP) return
        val newChamber = getChamberFromID(view.id)
        val currentChamber = userDiagnosis.chamberIdx


        if (newChamber >= 0 && currentChamber != newChamber) {
            if (currentChamber >= 0) clearChamberDiagnosis()
            userDiagnosis.chamberIdx = newChamber
        } else if (newChamber >= 0 && currentChamber == newChamber) {
            touchOnChamber(currentChamber, ix, iy)
        }
        setDiagramBackground()
        view.invalidate()
        Log.w(TAG, "onTouchEvent $currentChamber")
    }

    override fun onTouchEvent(view: com.ailab.aicardiotrainer.interpretation.InterpretationCanvasView, event: MotionEvent?, ix: Float, iy: Float) {
        TODO("Not yet implemented")
    }

    private fun clearChamberDiagnosis() {
        userDiagnosis.clearPoints()
    }

    override fun draw(view: DrawCanvasView, canvas: Canvas?) {
        val chamber = getChamberFromID(view.id)
        if (chamber >= 0 && chamber == userDiagnosis.chamberIdx) {
            val n = userDiagnosis.nPoints
            try {
                repeat(n) {
                    val point = userDiagnosis.getPoint(it)
                    val type = DicomDiagnosis.getPointType(point)
                    val color = getColorForAnomaly(type)

                    val screenPoint = view.getScreenCoordinate(point)
                    val paint = Paint()
                    paint.color = color
                    canvas?.drawCircle(screenPoint[0], screenPoint[1], 10.0F, paint)
                }
            } catch (e: Exception){
                Log.w(TAG, "draw: ${e}")
                userDiagnosis.points = JSONArray()
            }
        }
    }

    override fun draw(view: com.ailab.aicardiotrainer.interpretation.InterpretationCanvasView, canvas: Canvas?) {
        TODO("Not yet implemented")
    }

    private fun getColorForAnomaly(type: Int): Int {
        return when (type) {
            DicomDiagnosis.ANOMALY_REDUCED_ACTIVITY -> Color.RED
            DicomDiagnosis.ANOMALY_NO_ACTIVITY -> Color.GREEN
            DicomDiagnosis.ANOMALY_TWISTED_ACTIVITY -> Color.BLUE
            DicomDiagnosis.ANOMALY_NOT_SYNC_ACTIVITY -> Color.BLACK
            else -> Color.MAGENTA
        }
    }

    private fun setDiagramBackground() {
        iv_diagnosis_2C.setBackgroundColor(if(userDiagnosis.chamberIdx==0) Color.RED else Color.WHITE)
        iv_diagnosis_3C.setBackgroundColor(if(userDiagnosis.chamberIdx==1) Color.RED else Color.WHITE)
        iv_diagnosis_4C.setBackgroundColor(if(userDiagnosis.chamberIdx==2) Color.RED else Color.WHITE)
        iv_diagnosis_pts_l.setBackgroundColor(if(userDiagnosis.chamberIdx==3) Color.RED else Color.WHITE)
        iv_diagnosis_pts_s.setBackgroundColor(if(userDiagnosis.chamberIdx==4) Color.RED else Color.WHITE)
        iv_diagnosis_no_label.setBackgroundColor(if(userDiagnosis.chamberIdx==5) Color.RED else Color.WHITE)
    }

    private fun getChamberFromID(id: Int): Int {
        return when (id) {
            R.id.iv_diagnosis_2C -> 0
            R.id.iv_diagnosis_3C -> 1
            R.id.iv_diagnosis_4C -> 2
            R.id.iv_diagnosis_pts_l -> 3
            R.id.iv_diagnosis_pts_s -> 4
            R.id.iv_diagnosis_no_label -> 5
            else -> -1
        }
    }

    private fun touchOnChamber(chamber: Int, ix: Float, iy: Float) {
        if (chamber >= 0) {
            val atype = getCurrentAnomalyType()

            if (atype != -1) {
                userDiagnosis.addPoint(ix, iy, atype)
            } else {
                Toast.makeText(context, "Please choose anomaly type", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun getCurrentAnomalyType(): Int {
        if (rb_wall_reduced_activity.isChecked) return DicomDiagnosis.ANOMALY_REDUCED_ACTIVITY
        if (rb_wall_no_activity.isChecked) return DicomDiagnosis.ANOMALY_NO_ACTIVITY
        if (rb_wall_twisted_activity.isChecked) return DicomDiagnosis.ANOMALY_TWISTED_ACTIVITY
        if (rb_not_sync_activity.isChecked) return DicomDiagnosis.ANOMALY_NOT_SYNC_ACTIVITY
        return DicomDiagnosis.NORMAL_ACTIVITY
    }

}