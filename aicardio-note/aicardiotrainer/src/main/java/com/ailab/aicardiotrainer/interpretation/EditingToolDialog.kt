/*
 * Copyright 2021 UET-AILAB
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

package com.ailab.aicardiotrainer.interpretation

import android.app.Dialog
import android.content.Context
import android.graphics.Rect
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.Window
import com.ailab.aicardiotrainer.R
import kotlinx.android.synthetic.main.dialog_editing_tool.*

class EditingToolDialog(val activity: InterpretationActivity, val toolUsingListener: OnToolUsing): Dialog(activity) {
    companion object {
        const val TAG = "EditingToolDialog"

    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val displayRectangle = Rect()
        val window: Window = activity.window
        window.decorView.getWindowVisibleDisplayFrame(displayRectangle)

        // inflate and adjust layout
        val inflater : LayoutInflater = activity.getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater
        val layout: View = inflater.inflate(R.layout.dialog_editing_tool, null)
//        layout.minimumWidth = (displayRectangle.width() * 0.9f).toInt()
//        layout.minimumHeight = (displayRectangle.height() * 0.4f).toInt()
        layout.minimumWidth = 250
        layout.minimumHeight = 60
        setContentView(layout)

        bt_cancel.setOnClickListener { cancel() }
        bt_save.setOnClickListener { onSaveClicked() }

//        bt_tool_contrast.setOnClickListener {
//            onToolClicked("bt_tool_contrast", R.id.bt_tool_contrast)
//            toolUsingListener.onToolSelected("")
//        }

        bt_tool_draw_point.setOnClickListener {
            toolUsingListener.onToolSelected(InterpretationActVM.TOOL_DRAW_POINT, toolTypeClick = false)
        }

        bt_tool_draw_point.setOnLongClickListener {
            toolUsingListener.onToolSelected(InterpretationActVM.TOOL_DRAW_POINT, toolTypeClick = true);
            true
        }

        bt_tool_draw_boundary.setOnClickListener {
            toolUsingListener.onToolSelected(InterpretationActVM.TOOL_DRAW_BOUNDARY, toolTypeClick = false)
        }

        bt_tool_draw_boundary.setOnLongClickListener {
            toolUsingListener.onToolSelected(InterpretationActVM.TOOL_DRAW_BOUNDARY, toolTypeClick = true);
            true
        }

        bt_tool_measure_length.setOnClickListener {
            toolUsingListener.onToolSelected(InterpretationActVM.TOOL_MEASURE_LENGTH, toolTypeClick = false)
        }

        bt_tool_measure_area.setOnClickListener {
            toolUsingListener.onToolSelected(InterpretationActVM.TOOL_MEASURE_AREA, toolTypeClick = false)
        }

        bt_no_action.setOnClickListener {
            toolUsingListener.onToolSelected(InterpretationActVM.TOOL_NO_ACTION, toolTypeClick = false)

        }


    }

    private fun onSaveClicked() {
//        listener.onSaveConfirmed(file, dicomAnnotation, dicomDiagnosis)
        dismiss()
    }

//
//    fun onToolClicked(toolName: String, toolButtonID: Int) {
//        toolUsingListener.onToolSelected(toolName = toolName,  toolButtonID = toolButtonID)
//    }


}