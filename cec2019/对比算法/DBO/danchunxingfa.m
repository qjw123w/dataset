function X = danchunxingfa(Xg,Xb,Xs,fg,fs,dim,FF,fhd,fobj)
    Xc=(Xg+Xb)/2;%中心点
    Xr=Xc+1*(Xc-Xs);%反射点       Xs是差的点
    fr=feval(fhd,Xr',fobj)-FF;
%     fr=cec13(Xr',Dim,func_num,O,M,shuff );
    if fr<fg
        Xe=Xc+2*(Xr-Xc);%扩张点
        if feval(fhd,Xe',fobj)-FF<fg
%         if  cec13(Xe',Dim,func_num,O,M,shuff )<fg
           Xs=Xe;
        else
           Xs=Xr;
        end
    end
    if fr>fs
        Xt=Xc+0.5*(Xs-Xc);%压缩点
        if feval(fhd,Xt',fobj)-FF<fs
%         if cec13(Xt',Dim,func_num,O,M,shuff )<fs
               Xs=Xt;
        end
    end
    if  fr>fg&&fr<fs
        Xw=Xc-0.5*(Xs-Xc);
        if feval(fhd,Xw',fobj)-FF<fs
%         if cec13(Xw',Dim,func_num,O,M,shuff )<fs
           Xs=Xw;
        else
           Xs=Xr;
        end
    end
    X=Xs;
end

