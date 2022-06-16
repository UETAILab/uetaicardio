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

package com.ailab.aicardio.annotationscreen.views

import android.annotation.SuppressLint
import android.app.Activity
import android.app.Dialog
import android.content.Context
import android.graphics.*
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.Window
import android.widget.CompoundButton
import android.widget.Toast
import androidx.core.content.ContextCompat
import com.ailab.aicardio.R
import com.ailab.aicardio.annotationscreen.AnnotationActVM
import com.ailab.aicardio.annotationscreen.RenderDrawMVI
import com.ailab.aicardio.annotationscreen.interfaces.OnDrawListener
import com.ailab.aicardio.annotationscreen.interfaces.OnNormalizeTouchListener
import com.ailab.aicardio.annotationscreen.views.RadioGridGroup.*
import com.ailab.aicardio.repository.DicomDiagnosis
import com.ailab.aicardio.repository.Polygon
import kotlinx.android.synthetic.main.dialog_diagnosis.*
import org.json.JSONArray
import org.json.JSONObject

class DiagnosisDialog(
    val activity: Activity,
    private val listener: OnDiagnosisEnteredListener,
    dicomDiagnosis: DicomDiagnosis,
    val annotationVM: AnnotationActVM
) : Dialog(activity), CompoundButton.OnCheckedChangeListener,
    OnNormalizeTouchListener,
    OnDrawListener {
    // https://quizlet.com/kr/377954294/%EC%88%9C%ED%99%98%EA%B8%B0-flash-cards/

    companion object {
        val TAG = "DiagnosisDialog"
        var bitmaps = HashMap<Int, Bitmap>()
        var bitmapDefault : Bitmap? = null

        val indexHeart4C2C3C = arrayListOf<Pair<Int, Int> > (
            Pair(0, 11),
            Pair(12, 20),
            Pair(21, 30),
            Pair(31, 39),
            Pair(40, 47)
        )


    }

    val userDiagnosis = DicomDiagnosis(dicomDiagnosis.toString())

    interface OnDiagnosisEnteredListener {
        fun onDiagnosisEntered(dicomDiagnosis: DicomDiagnosis)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestWindowFeature(Window.FEATURE_NO_TITLE)

//        setContentView(R.layout.dialog_diagnosis)
        // inflate and adjust layout

        val displayRectangle = Rect()
        val window: Window = activity.window
        window.decorView.getWindowVisibleDisplayFrame(displayRectangle)

        val inflater : LayoutInflater = activity.getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater
        val layout: View = inflater.inflate(R.layout.dialog_diagnosis, null)
        layout.minimumWidth = (displayRectangle.width() * 0.9f).toInt()
//        layout.minimumHeight = (displayRectangle.height() * 0.8f).toInt()

        setContentView(layout)
//        Log.w(TAG, "Size: W ${layout.width} H ${layout.height} ")

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


        radioGrid.setOnCheckedChangeListener(object : OnCheckedChangeListener {
            override fun onCheckedChanged(group: RadioGridGroup?, checkedId: Int) {

                when (checkedId) {
                    R.id.rb_wall_hyperactivity -> {
                        Log.w(TAG, "radioGrid id check: rb_wall_hyperactivity")
                    }

                    R.id.rb_wall_normal_motion -> {
                        Log.w(TAG, "radioGrid id check: rb_wall_normal_motion")
                    }
                    R.id.rb_wall_post_systolic_contraction -> {
                        Log.w(TAG, "radioGrid id check: rb_wall_post_systolic_contraction")
                    }
                    R.id.rb_wall_hypokinetic_motion -> {
                        Log.w(TAG, "radioGrid id check: rb_wall_hypokinetic_motion")
                    }
                    R.id.rb_wall_akinetic_motion -> {
                        Log.w(TAG, "radioGrid id check: rb_wall_akinetic_motion")
                    }
                    R.id.rb_wall_dyskinetic_motion -> {
                        Log.w(TAG, "radioGrid id check: rb_wall_dyskinetic_motion")
                    }

                    R.id.rb_wall_paradoxical_motion -> {
                        Log.w(TAG, "radioGrid id check: rb_wall_paradoxical_motion")
                    }
                    R.id.rb_wall_dyssynchronized_motion -> {
                        Log.w(TAG, "radioGrid id check: rb_wall_dyssynchronized_motion")
                    }
                    else -> {
                        Log.w(TAG, "None clicked")
                    }

                }

            }
        })

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
                            R.id.iv_diagnosis_2C -> R.drawable.label_heart_2c
                            R.id.iv_diagnosis_3C -> R.drawable.label_heart_3c
                            R.id.iv_diagnosis_4C -> R.drawable.label_heart_4c
                            R.id.iv_diagnosis_pts_l -> R.drawable.label_heart_pts_l
                            R.id.iv_diagnosis_pts_s -> R.drawable.label_heart_pts_s
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
                    paint.color = ContextCompat.getColor(view.context, color)

                    paint.strokeWidth = 3.0F
                    paint.alpha = 100
                    paint.textSize = 30F

                    canvas?.drawCircle(screenPoint[0], screenPoint[1], 5.0F, paint)
//                    0 -> "2C"
//                    1 -> "3C"
//                    2 -> "4C"
//                    3 -> "PST_L"
//                    4 -> "PST_S"

                    val indexSector = when (chamber) {
                        0 -> indexHeart4C2C3C[1] // 2C
                        1 -> indexHeart4C2C3C[2] // 3C
                        2 -> indexHeart4C2C3C[0] // 4C
                        3 -> indexHeart4C2C3C[3] // PST_L
                        4 -> indexHeart4C2C3C[4] // PST_S
                        else -> Pair(-1, -1)
                    }

                    val x = (point.getDouble("x") * 250).toInt()
                    val y = (point.getDouble("y") * 300).toInt()


                    val indexDrawSector = getIndexOfHeartSector(annotationVM.boundaryHeart, Point(x, y)
                    , indexSector )
                    if (indexDrawSector != -1) {
                        val heartBoundary = annotationVM.boundaryHeart.getJSONObject(indexDrawSector).getJSONArray("boundary")
                        RenderDrawMVI.drawPolygon(view, canvas, heartBoundary, paint)
                    }
                }
            } catch (e: Exception){
                Log.w(TAG, "draw: ${e}")
                userDiagnosis.points = JSONArray()
            }
        }
    }

    fun getIndexOfHeartSector(boundaryHeart: JSONArray, point: Point, indexSector: Pair<Int, Int> ): Int {
//        for
//        Log.w(TAG, "${point} ${indexSector}" )

        for (i in indexSector.first..indexSector.second) {
            try {
                val boundaryIth: JSONObject = boundaryHeart.getJSONObject(i)
                val convexHull = boundaryIth.getJSONArray("convex_hull_int").getJSONArray(0)

                val xp = arrayListOf<Int>()
                val yp = arrayListOf<Int>()

                repeat(convexHull.length()){

                    val pc = convexHull.getJSONObject(it)
                    val x = pc.getInt("x")
                    val y = pc.getInt("y")
                    xp.add(x)
                    yp.add(y)
                }

                val checkInside = checkInsidePolygon(convexHull.length(), xp, yp, point.x, point.y)
                if (checkInside == 1) return i
            } catch(e: Exception) {
                Log.w(TAG, "getIndexOfHeartSector: ${e}")
            }
        }

        return -1
    }


    fun checkInsidePolygon(npol: Int, xp: ArrayList<Int>, yp: ArrayList<Int>, x: Int, y: Int ): Int {
        var i = 0
        var j = npol - 1
        var c = 0
        while (i < npol) {
            if ((((yp[i] <= y) && (y < yp[j])) || ((yp[j] <= y) && (y < yp[i]))) &&
                    (x < (xp[j] - xp[i]) * (y - yp[i]) / (yp[j] - yp[i]) + xp[i]))
                    c = 1 - c
            j = i
            i += 1
        }
        return c
    }

    private fun getColorForAnomaly(type: Int): Int {
//        Log.w(TAG, "getColorForAnomaly ${type}")
        return when (type) {
            DicomDiagnosis.ANORMAL_hyperactivity -> R.color.colorOrange
            DicomDiagnosis.NORMAL_motion -> R.color.colorGrey
            DicomDiagnosis.ANORMAL_post_systolic_contraction -> R.color.colorPurple
            DicomDiagnosis.ANORMAL_hypokinetic_motion -> R.color.colorGreen
            DicomDiagnosis.ANORMAL_akinetic_motion -> R.color.colorYellow
            DicomDiagnosis.ANORMAL_dyskinetic_motion -> R.color.colorRed
            DicomDiagnosis.ANORMAL_paradoxical_motion -> R.color.colorBrown
            DicomDiagnosis.ANORMAL_dyssynchronized_motion -> R.color.colorBlack
            else -> R.color.colorRED
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

        if (rb_wall_hyperactivity.isChecked) return DicomDiagnosis.ANORMAL_hyperactivity
        if (rb_wall_normal_motion.isChecked) return DicomDiagnosis.NORMAL_motion
        if (rb_wall_post_systolic_contraction.isChecked) return DicomDiagnosis.ANORMAL_post_systolic_contraction
        if (rb_wall_hypokinetic_motion.isChecked) return DicomDiagnosis.ANORMAL_hypokinetic_motion
        if (rb_wall_akinetic_motion.isChecked) return DicomDiagnosis.ANORMAL_akinetic_motion
        if (rb_wall_dyskinetic_motion.isChecked) return DicomDiagnosis.ANORMAL_dyskinetic_motion
        if (rb_wall_paradoxical_motion.isChecked) return DicomDiagnosis.ANORMAL_paradoxical_motion
        if (rb_wall_dyssynchronized_motion.isChecked) return DicomDiagnosis.ANORMAL_dyssynchronized_motion

        return -1
    }

}