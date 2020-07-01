package com.dan.recture.service;

import java.util.List;

import javax.inject.Inject;

import org.springframework.stereotype.Service;

import com.dan.recture.domain.AirbnbResultVO;
import com.dan.recture.persistance.AirbnbResultDAO;


@Service
public class AirbnbResultServiceImpl implements AirbnbResultService {
	
	@Inject
	AirbnbResultDAO dao;

	@Override
	public List<AirbnbResultVO> selectResult() throws Exception {
		// TODO Auto-generated method stub
		return dao.selectResult();
	}

}
