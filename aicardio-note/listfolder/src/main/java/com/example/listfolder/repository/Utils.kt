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

package com.example.listfolder.repository

import android.app.Activity
import android.content.pm.PackageManager
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.File
import java.util.logging.Logger

class Utils {

    companion object {
        val TAG = "Utils"

        val DEFAULT_FOLDER_DOWNLOAD = "/storage/emulated/0/Download"

        private const val MY_PERMISSIONS_REQUEST_CODE = 1

        fun walk(file: File?, arrayFolder: ArrayList<String>, compact: Boolean){

            Logger.getLogger(TAG).warning("go to walk")
            var newFile = file?: File(DEFAULT_FOLDER_DOWNLOAD)

            newFile.listFiles()?.forEach {pathFileInFolder ->
                val path = pathFileInFolder.absolutePath
                Logger.getLogger(TAG).warning("$path")

                if (compact) {
                    if (pathFileInFolder.isFile && !path.contains(".json")){
                        arrayFolder.add(path)
                    }else if (pathFileInFolder.isDirectory){
                        walk(pathFileInFolder, arrayFolder, compact)
                    }
                } else {
                    if (!path.contains(".json")) {
                        arrayFolder.add(path)
                    }
                }
            }
        }


        fun checkAndRequestPermissions(
            activity: Activity,
            permissions: Array<String>
        ): Boolean {
            // Here, thisActivity is the current activity
            var granted = true
            permissions.forEach { permission ->
                if (ContextCompat.checkSelfPermission(
                        activity,
                        permission
                    ) != PackageManager.PERMISSION_GRANTED
                )
                    granted = false
            }

            val permissionGranted = if (!granted) {
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
            return permissionGranted
//        log.warning("Permission granted = $permissionGranted")
        }
    }


}