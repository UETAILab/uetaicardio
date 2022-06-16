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

package com.uetailab.aipacs.home_pacs

import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.util.Log
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

fun Context.toast(message: String, length: Int = Toast.LENGTH_SHORT) {
    Toast.makeText(this, message, length).show()
}
//LCE -> Loading/Content/Error
sealed class LCE<out T> {
    data class Result<T>(val data: T, val error: Boolean = true, val message: String = "error") : LCE<T>()
 }

fun checkAndRequestPermissions(activity: Activity, permissions: Array<String>, MY_PERMISSIONS_REQUEST_CODE: Int = 1): Boolean {
    // Here, thisActivity is the current activity
    var granted = true
    permissions.forEach { permission ->
        if (ContextCompat.checkSelfPermission(
                activity,
                permission
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            granted = false
            Log.w("checkAndRequestPermissions", " NOT granted $permission")
        }
        else {
            Log.w("checkAndRequestPermissions", " granted $permission")
        }
    }

    return if (!granted) {
        // Permission is not granted
        // Should we show an explanation?
        ActivityCompat.requestPermissions(
            activity,
            permissions,
            MY_PERMISSIONS_REQUEST_CODE
        )
        false
    } else {
        true
    }
//        log.warning("Permission granted = $permissionGranted")
}