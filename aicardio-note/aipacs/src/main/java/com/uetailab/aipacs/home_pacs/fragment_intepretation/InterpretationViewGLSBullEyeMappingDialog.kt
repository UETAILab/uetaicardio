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

package com.uetailab.aipacs.home_pacs.fragment_intepretation


import android.app.Activity
import android.app.Dialog
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.Window
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.github.mikephil.charting.interfaces.datasets.ILineDataSet
import com.github.mikephil.charting.utils.ColorTemplate
import com.uetailab.aipacs.R
import kotlinx.android.synthetic.main.interpretation_view_dialog_gls_bull_eye_mapping.*
import org.json.JSONObject
import kotlin.random.Random


class InterpretationViewGLSBullEyeMappingDialog(
    val activity: Activity,
    val interpretationViewVM: InterpretationViewVM,
    val bitmap: Bitmap // bull_eye_mapping.png
) : Dialog(activity) {
    // example: https://github.com/PhilJay/MPAndroidChart/blob/master/MPChartExample/src/main/java/com/xxmassdeveloper/mpchartexample/MultiLineChartActivity.java

    companion object {
        val TAG = "InterpretationViewGLSBullEyeMappingDialog"

        fun getPoint(x: Double, y: Double): JSONObject {
            val r = JSONObject()
            r.put("x", x)
            r.put("y", y)
            return r
        }
        // kich thuoc cua anh bull_eye (724, 801, 3) (h, w, c) -> toa do x duoc chia cho 801

        val POSITION_DRAW_GLS = arrayListOf<JSONObject>(
            getPoint(0.49313358302122345, 0.5939226519337016),
            getPoint(0.4431960049937578, 0.6284530386740331),
            getPoint(0.38701622971285893, 0.5939226519337016),
            getPoint(0.38701622971285893, 0.5248618784530387),
            getPoint(0.4431960049937578, 0.4903314917127072),
            getPoint(0.49313358302122345, 0.5248618784530387),
            getPoint(0.6054931335830213, 0.6629834254143646),
            getPoint(0.4431960049937578, 0.7665745856353591),
            getPoint(0.2808988764044944, 0.6629834254143646),
            getPoint(0.2808988764044944, 0.4558011049723757),
            getPoint(0.4431960049937578, 0.35220994475138123),
            getPoint(0.599250936329588, 0.4558011049723757),
            getPoint(0.7116104868913857, 0.7320441988950276),
            getPoint(0.4431960049937578, 0.9046961325966851),
            getPoint(0.16853932584269662, 0.7320441988950276),
            getPoint(0.16853932584269662, 0.3867403314917127),
            getPoint(0.4431960049937578, 0.21408839779005526),
            getPoint(0.7116104868913857, 0.3867403314917127)
        )
        const val NUM_SEGMENTATION_MODEL = 18
    }


    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)

        val displayRectangle = Rect()
        val window: Window = activity.window
        window.decorView.getWindowVisibleDisplayFrame(displayRectangle)

        // inflate and adjust layout
        val inflater : LayoutInflater = activity.getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater
        val layout: View = inflater.inflate(R.layout.interpretation_view_dialog_gls_bull_eye_mapping, null)
//
        layout.minimumWidth = displayRectangle.width()
        layout.minimumHeight = displayRectangle.height()

        setContentView(layout)

        bt_cancel_dialog_gls_bull_eye_mapping.setOnClickListener {
            cancel()
        }

        bt_ok_dialog_gls_bull_eye_mapping.setOnClickListener {
            dismiss()
        }



        iv_draw_canvas_view_bull_eye_mapping.setFitScale(bitmap)
        iv_draw_canvas_view_bull_eye_mapping.setCustomImageBitmap(bitmap)

        iv_draw_canvas_view_bull_eye_mapping.setOnDrawListener(object : OnDrawListener{
            override fun draw(view: InterpretationViewStudyPreviewCanvasView, canvas: Canvas?) {

                drawBullEyeMapping(view, canvas)
//                interpretationViewVM.viewStates().value?.let {
//
////                    drawCanvasEFCalculationDialogByType(it, interpretationViewVM, indexEDV, view, canvas)
//                }
            }
        })



    }

    fun drawBullEyeMapping(view: InterpretationViewStudyPreviewCanvasView, canvas: Canvas?) {
//        Log.w(TAG, "onDraw InterpretationCanvasView")

        fun getRandomList(numFrame: Int, random: Random): List<Float> =
            List(numFrame) { random.nextDouble(-22.0, -14.0).toFloat() }

        val gls_array = getRandomList(18, Random(2))

        repeat(NUM_SEGMENTATION_MODEL) {

            val text = gls_array.get(it).toLong().toString()
//            Log.w(TAG, "id: ${it} text: ${text}")
            val p = POSITION_DRAW_GLS.get(it)
            val ps = view.getScreenCoordinate(p)
            view.drawTextAtPosition(canvas, ps[0], ps[1], text)

        }
    }


}