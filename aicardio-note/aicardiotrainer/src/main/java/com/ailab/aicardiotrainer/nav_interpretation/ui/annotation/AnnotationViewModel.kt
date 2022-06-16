package com.ailab.aicardiotrainer.nav_interpretation.ui.annotation

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel

class AnnotationViewModel : ViewModel() {

    private val _text = MutableLiveData<String>().apply {
        value = "This is annotation Fragment"
    }
    val text: LiveData<String> = _text
}