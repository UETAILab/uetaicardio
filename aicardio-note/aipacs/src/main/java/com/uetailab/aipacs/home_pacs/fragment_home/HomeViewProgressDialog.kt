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

package com.uetailab.aipacs.home_pacs.fragment_home

import android.app.Activity
import android.app.AlertDialog
import android.app.Dialog
import android.content.Context
import android.graphics.Rect
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.Window
import com.uetailab.aipacs.R
import kotlinx.android.synthetic.main.fragment_home_view_dialog_progress.*

class HomeViewProgressDialog(val activity: Activity) : Dialog(activity) {
    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)

        val displayRectangle = Rect()
        val window: Window = activity.window
        window.decorView.getWindowVisibleDisplayFrame(displayRectangle)

        // inflate and adjust layout
        val inflater : LayoutInflater = activity.getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater
        val layout: View = inflater.inflate(R.layout.fragment_home_view_dialog_progress, null)
        layout.minimumWidth = (displayRectangle.width() * 0.9f).toInt()
//        layout.minimumHeight = (displayRectangle.height() * 0.4f).toInt()
        setContentView(layout)

        pb_progress.max = 100
    }
}