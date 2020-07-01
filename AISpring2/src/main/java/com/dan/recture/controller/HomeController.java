package com.dan.recture.controller;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.DateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.UUID;

import javax.inject.Inject;

import org.python.util.PythonInterpreter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.multipart.MultipartFile;

import com.dan.recture.domain.AirbnbResultVO;
import com.dan.recture.domain.TextAnalysisModel;
import com.dan.recture.service.TextAnalysisService;


/**
 * Handles requests for the application home page.
 */
@Controller
public class HomeController {
	
	private static final Logger logger = LoggerFactory.getLogger(HomeController.class);
	
	/**
	 * Simply selects the home view to render by returning its name.
	 */
	@RequestMapping(value = "/", method = RequestMethod.GET)
	public String home(Locale locale, Model model) {
		logger.info("Welcome home! The client locale is {}.", locale);
		
		Date date = new Date();
		DateFormat dateFormat = DateFormat.getDateTimeInstance(DateFormat.LONG, DateFormat.LONG, locale);
		
		String formattedDate = dateFormat.format(date);
		
		model.addAttribute("serverTime", formattedDate );
		
		return "home";
	}
	
	//현진
	@RequestMapping(value = "/airbnbPrice")
	public String viewHome() {
		return "home2";
	}
	
	//사용자별 history 관리 시
	//String Name을 이 위치에 선언해 전역변수로 사용하고 python실행시 넘기는것도 좋은방법
	@RequestMapping(value="/fileupload",method=RequestMethod.GET)
	public String fileuploadGET()throws Exception{
		return "fileupload";
	}
	@RequestMapping(value = "/fileupload",method = RequestMethod.POST)
	public void upload(MultipartFile uploadfile, Model model){
	    logger.info("upload() POST 호출");
	    logger.info("파일 이름: {}", uploadfile.getOriginalFilename());
	    logger.info("파일 크기: {}", uploadfile.getSize());

	    String Name = saveFile(uploadfile);
	    model.addAttribute("fileName", Name);
	    
	}
	private String saveFile(MultipartFile file){
	    // 파일 이름 변경
	    //UUID uuid = UUID.randomUUID();
	    //String RealName = uuid + "_" + file.getOriginalFilename();
	    String RealName = file.getOriginalFilename();
	    String saveName = "uploadFile.csv";

	    logger.info("saveName: {}",saveName);

	    // 저장할 File 객체를 생성(껍데기 파일)ㄴ
	    String UPLOAD_PATH="/Users/noyeongdan/Downloads/Server_AI/AISpring2/src/main/webapp/WEB-INF/views";
	    File saveFile = new File(UPLOAD_PATH, saveName); // 저장할 폴더 이름, 저장할 파일 이름
	    
	    try {
	        file.transferTo(saveFile); // 업로드 파일에 saveFile이라는 껍데기 입힘
	    } catch (IOException e) {
	        e.printStackTrace();
	        return null;
	    }
	    
	    return RealName;
	}
	
	@Inject
	private TextAnalysisService service;
//	@RequestMapping(value="/Analysis",method=RequestMethod.GET)
//	public void listGET(TextAnalysisModel result, Model model)throws Exception{
//		model.addAttribute("list", service.list(result));
//	}
	@RequestMapping(value="/Analysis",method=RequestMethod.GET)
	public void listGET(Model model)throws Exception{
		List<TextAnalysisModel> list=service.Result();
		model.addAttribute("list", list);
	}
	
	@RequestMapping(value="/Analysis",method=RequestMethod.POST)
	public void AnalysisPOST(Model model)throws Exception{
		Python();
//		model.addAttribute("list", service.list(result));
		model.addAttribute("list", service.Result());
	}
	private void Python()throws Exception{
		
		BufferedReader input =null;
	     
	    try {
	        long start, end;
	        String line;
	        String execPath ="python //Users//noyeongdan//Downloads//Server_AI//AISpring2//src//main//webapp//WEB-INF//views//main.py";
	         
	        start = System.currentTimeMillis();
	         
	        Process p = Runtime.getRuntime().exec(execPath);
	        input =new BufferedReader(new InputStreamReader(p.getInputStream()));
	        
	        line = input.readLine();
	        logger.info(line);
	        while ((line = input.readLine()) !=null) {
	            logger.info("Result : " + line);
	        }
	 
	        end = System.currentTimeMillis();
	 
	        logger.info("Running Time : " + (end - start) / 1000f +"s.");
	        //model.addAttribute("result", "true");
	         
	    }catch (IOException err) {
	        err.printStackTrace();
	    }finally {
	        if (input !=null) input.close();
	    }
	    
//		System.setProperty("python.cachedir.skip", "true");
//      PythonInterpreter interpreter = new PythonInterpreter();

//      interpreter.execfile("/Users/noyeongdan/Downloads/Spring4/AISpring/src/main/webapp/WEB-INF/views/test.py");
//      interpreter.exec("addition(1,1)");
      
//      interpreter.execfile("/Users/noyeongdan/Downloads/Spring4/AISpring/src/main/webapp/WEB-INF/views/test2.py");
//      
//      interpreter.exec("import scipy.io");
//      interpreter.exec("import csv");
//      interpreter.exec("import pymysql");
//      interpreter.exec("import pandas as pd");
//      interpreter.exec("import MeCab, csv, os");
//      interpreter.exec("import glob, pandas as pd, numpy as np");
//      interpreter.exec("from sqlalchemy import create_engine");
//      interpreter.exec("import os, json, glob, sys, numpy as np");
//      interpreter.exec("import matplotlib.pyplot as plt");
//      interpreter.exec("import matplotlib as mpl");
//      interpreter.exec("import keras.backend.tensorflow_backend as K");
//      interpreter.exec("import tensorflow as tf");
//      interpreter.exec("from keras.preprocessing.text import Tokenizer");
//      interpreter.exec("from keras.preprocessing.sequence import pad_sequences");
//      interpreter.exec("from keras.preprocessing import sequence");
//      interpreter.exec("from keras.models import Sequential");
//      interpreter.exec("from keras.layers import Dense, Embedding, LSTM, Flatten, Dropout, Input, Conv1D, MaxPooling1D, GlobalMaxPool1D");
//      interpreter.exec("from keras.utils import np_utils");
//      interpreter.exec("from keras.callbacks import EarlyStopping, ModelCheckpoint");
//      interpreter.exec("from sklearn.model_selection import train_test_split");
//      interpreter.exec("from sqlalchemy import create_engine");
//      
//      
//      
//      interpreter.exec("fileUpload()");
//      interpreter.exec("file_to_ids()");
//      interpreter.exec("save()");
//      interpreter.exec("Model()");

	}
}
