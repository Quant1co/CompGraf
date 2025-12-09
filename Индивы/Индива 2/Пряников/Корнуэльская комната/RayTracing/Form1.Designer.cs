namespace RayTracing
{
    partial class Form1
    {
        /// <summary>
        /// Обязательная переменная конструктора.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Освободить все используемые ресурсы.
        /// </summary>
        /// <param name="disposing">истинно, если управляемый ресурс должен быть удален; иначе ложно.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Код, автоматически созданный конструктором форм Windows

        /// <summary>
        /// Требуемый метод для поддержки конструктора — не изменяйте 
        /// содержимое этого метода с помощью редактора кода.
        /// </summary>
        private void InitializeComponent()
        {
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.button1 = new System.Windows.Forms.Button();
            this.cubeSpecularCB = new System.Windows.Forms.CheckBox();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.sphereSpecularCB = new System.Windows.Forms.CheckBox();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.refractSphereCB = new System.Windows.Forms.CheckBox();
            this.refractCubeCB = new System.Windows.Forms.CheckBox();
            this.twoLightsCB = new System.Windows.Forms.CheckBox();
            this.frontWallSpecularCB = new System.Windows.Forms.CheckBox();
            this.backWallSpecularCB = new System.Windows.Forms.CheckBox();
            this.leftWallSpecularCB = new System.Windows.Forms.CheckBox();
            this.rightWallSpecularCB = new System.Windows.Forms.CheckBox();
            this.upWallSpecularCB = new System.Windows.Forms.CheckBox();
            this.downWallSpecularCB = new System.Windows.Forms.CheckBox();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.groupBox4 = new System.Windows.Forms.GroupBox();
            this.statusLabel = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.groupBox3.SuspendLayout();
            this.groupBox4.SuspendLayout();
            this.SuspendLayout();
            // 
            // pictureBox1
            // 
            this.pictureBox1.BackColor = System.Drawing.SystemColors.MenuText;
            this.pictureBox1.Location = new System.Drawing.Point(3, 2);
            this.pictureBox1.Margin = new System.Windows.Forms.Padding(4);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(776, 683);
            this.pictureBox1.TabIndex = 0;
            this.pictureBox1.TabStop = false;
            // 
            // button1
            // 
            this.button1.BackColor = System.Drawing.Color.Honeydew;
            this.button1.FlatAppearance.BorderColor = System.Drawing.Color.FromArgb(((int)(((byte)(93)))), ((int)(((byte)(135)))), ((int)(((byte)(177)))));
            this.button1.FlatAppearance.MouseDownBackColor = System.Drawing.Color.FromArgb(((int)(((byte)(46)))), ((int)(((byte)(63)))), ((int)(((byte)(92)))));
            this.button1.FlatAppearance.MouseOverBackColor = System.Drawing.Color.FromArgb(((int)(((byte)(64)))), ((int)(((byte)(87)))), ((int)(((byte)(126)))));
            this.button1.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.button1.Font = new System.Drawing.Font("Microsoft Sans Serif", 19.875F, ((System.Drawing.FontStyle)((System.Drawing.FontStyle.Bold | System.Drawing.FontStyle.Italic))), System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.button1.ForeColor = System.Drawing.Color.Black;
            this.button1.Location = new System.Drawing.Point(787, 467);
            this.button1.Margin = new System.Windows.Forms.Padding(4);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(332, 196);
            this.button1.TabIndex = 1;
            this.button1.Text = "Нарисовать";
            this.button1.UseVisualStyleBackColor = false;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // cubeSpecularCB
            // 
            this.cubeSpecularCB.AutoSize = true;
            this.cubeSpecularCB.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.875F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.cubeSpecularCB.Location = new System.Drawing.Point(14, 66);
            this.cubeSpecularCB.Margin = new System.Windows.Forms.Padding(6);
            this.cubeSpecularCB.Name = "cubeSpecularCB";
            this.cubeSpecularCB.Size = new System.Drawing.Size(44, 17);
            this.cubeSpecularCB.TabIndex = 2;
            this.cubeSpecularCB.Text = "Куб";
            this.cubeSpecularCB.UseVisualStyleBackColor = true;
            // 
            // groupBox1
            // 
            this.groupBox1.BackColor = System.Drawing.SystemColors.Window;
            this.groupBox1.Controls.Add(this.sphereSpecularCB);
            this.groupBox1.Controls.Add(this.cubeSpecularCB);
            this.groupBox1.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.875F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.groupBox1.Location = new System.Drawing.Point(957, 16);
            this.groupBox1.Margin = new System.Windows.Forms.Padding(6);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Padding = new System.Windows.Forms.Padding(6);
            this.groupBox1.Size = new System.Drawing.Size(165, 171);
            this.groupBox1.TabIndex = 3;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Зеркальность фигур";
            this.groupBox1.Enter += new System.EventHandler(this.groupBox1_Enter);
            // 
            // sphereSpecularCB
            // 
            this.sphereSpecularCB.AutoSize = true;
            this.sphereSpecularCB.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.875F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.sphereSpecularCB.Location = new System.Drawing.Point(14, 110);
            this.sphereSpecularCB.Margin = new System.Windows.Forms.Padding(6);
            this.sphereSpecularCB.Name = "sphereSpecularCB";
            this.sphereSpecularCB.Size = new System.Drawing.Size(47, 17);
            this.sphereSpecularCB.TabIndex = 2;
            this.sphereSpecularCB.Text = "Шар";
            this.sphereSpecularCB.UseVisualStyleBackColor = true;
            // 
            // groupBox2
            // 
            this.groupBox2.BackColor = System.Drawing.SystemColors.Window;
            this.groupBox2.Controls.Add(this.refractSphereCB);
            this.groupBox2.Controls.Add(this.refractCubeCB);
            this.groupBox2.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.875F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.groupBox2.Location = new System.Drawing.Point(957, 183);
            this.groupBox2.Margin = new System.Windows.Forms.Padding(6);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Padding = new System.Windows.Forms.Padding(6);
            this.groupBox2.Size = new System.Drawing.Size(165, 186);
            this.groupBox2.TabIndex = 3;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Прозрачность фигур";
            this.groupBox2.Enter += new System.EventHandler(this.groupBox2_Enter);
            // 
            // refractSphereCB
            // 
            this.refractSphereCB.AutoSize = true;
            this.refractSphereCB.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.875F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.refractSphereCB.Location = new System.Drawing.Point(14, 120);
            this.refractSphereCB.Margin = new System.Windows.Forms.Padding(6);
            this.refractSphereCB.Name = "refractSphereCB";
            this.refractSphereCB.Size = new System.Drawing.Size(47, 17);
            this.refractSphereCB.TabIndex = 2;
            this.refractSphereCB.Text = "Шар";
            this.refractSphereCB.UseVisualStyleBackColor = true;
            // 
            // refractCubeCB
            // 
            this.refractCubeCB.AutoSize = true;
            this.refractCubeCB.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.875F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.refractCubeCB.Location = new System.Drawing.Point(14, 76);
            this.refractCubeCB.Margin = new System.Windows.Forms.Padding(6);
            this.refractCubeCB.Name = "refractCubeCB";
            this.refractCubeCB.Size = new System.Drawing.Size(44, 17);
            this.refractCubeCB.TabIndex = 2;
            this.refractCubeCB.Text = "Куб";
            this.refractCubeCB.UseVisualStyleBackColor = true;
            // 
            // twoLightsCB
            // 
            this.twoLightsCB.AutoSize = true;
            this.twoLightsCB.BackColor = System.Drawing.SystemColors.Window;
            this.twoLightsCB.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.875F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.twoLightsCB.Location = new System.Drawing.Point(13, 39);
            this.twoLightsCB.Margin = new System.Windows.Forms.Padding(6);
            this.twoLightsCB.Name = "twoLightsCB";
            this.twoLightsCB.Size = new System.Drawing.Size(87, 17);
            this.twoLightsCB.TabIndex = 4;
            this.twoLightsCB.Text = "2 источника";
            this.twoLightsCB.UseVisualStyleBackColor = false;
            this.twoLightsCB.CheckedChanged += new System.EventHandler(this.twoLightsCB_CheckedChanged);
            // 
            // frontWallSpecularCB
            // 
            this.frontWallSpecularCB.AutoSize = true;
            this.frontWallSpecularCB.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.875F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.frontWallSpecularCB.Location = new System.Drawing.Point(14, 63);
            this.frontWallSpecularCB.Margin = new System.Windows.Forms.Padding(6);
            this.frontWallSpecularCB.Name = "frontWallSpecularCB";
            this.frontWallSpecularCB.Size = new System.Drawing.Size(76, 17);
            this.frontWallSpecularCB.TabIndex = 0;
            this.frontWallSpecularCB.Text = "Передняя";
            this.frontWallSpecularCB.UseVisualStyleBackColor = true;
            // 
            // backWallSpecularCB
            // 
            this.backWallSpecularCB.AutoSize = true;
            this.backWallSpecularCB.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.875F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.backWallSpecularCB.Location = new System.Drawing.Point(14, 110);
            this.backWallSpecularCB.Margin = new System.Windows.Forms.Padding(6);
            this.backWallSpecularCB.Name = "backWallSpecularCB";
            this.backWallSpecularCB.Size = new System.Drawing.Size(63, 17);
            this.backWallSpecularCB.TabIndex = 0;
            this.backWallSpecularCB.Text = "Задняя";
            this.backWallSpecularCB.UseVisualStyleBackColor = true;
            // 
            // leftWallSpecularCB
            // 
            this.leftWallSpecularCB.AutoSize = true;
            this.leftWallSpecularCB.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.875F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.leftWallSpecularCB.Location = new System.Drawing.Point(14, 158);
            this.leftWallSpecularCB.Margin = new System.Windows.Forms.Padding(6);
            this.leftWallSpecularCB.Name = "leftWallSpecularCB";
            this.leftWallSpecularCB.Size = new System.Drawing.Size(58, 17);
            this.leftWallSpecularCB.TabIndex = 0;
            this.leftWallSpecularCB.Text = "Левая";
            this.leftWallSpecularCB.UseVisualStyleBackColor = true;
            // 
            // rightWallSpecularCB
            // 
            this.rightWallSpecularCB.AutoSize = true;
            this.rightWallSpecularCB.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.875F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.rightWallSpecularCB.Location = new System.Drawing.Point(14, 205);
            this.rightWallSpecularCB.Margin = new System.Windows.Forms.Padding(6);
            this.rightWallSpecularCB.Name = "rightWallSpecularCB";
            this.rightWallSpecularCB.Size = new System.Drawing.Size(64, 17);
            this.rightWallSpecularCB.TabIndex = 0;
            this.rightWallSpecularCB.Text = "Правая";
            this.rightWallSpecularCB.UseVisualStyleBackColor = true;
            // 
            // upWallSpecularCB
            // 
            this.upWallSpecularCB.AutoSize = true;
            this.upWallSpecularCB.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.875F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.upWallSpecularCB.Location = new System.Drawing.Point(14, 254);
            this.upWallSpecularCB.Margin = new System.Windows.Forms.Padding(6);
            this.upWallSpecularCB.Name = "upWallSpecularCB";
            this.upWallSpecularCB.Size = new System.Drawing.Size(68, 17);
            this.upWallSpecularCB.TabIndex = 0;
            this.upWallSpecularCB.Text = "Верхняя";
            this.upWallSpecularCB.UseVisualStyleBackColor = true;
            // 
            // downWallSpecularCB
            // 
            this.downWallSpecularCB.AutoSize = true;
            this.downWallSpecularCB.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.875F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.downWallSpecularCB.Location = new System.Drawing.Point(13, 308);
            this.downWallSpecularCB.Margin = new System.Windows.Forms.Padding(6);
            this.downWallSpecularCB.Name = "downWallSpecularCB";
            this.downWallSpecularCB.Size = new System.Drawing.Size(66, 17);
            this.downWallSpecularCB.TabIndex = 0;
            this.downWallSpecularCB.Text = "Нижняя";
            this.downWallSpecularCB.UseVisualStyleBackColor = true;
            // 
            // groupBox3
            // 
            this.groupBox3.BackColor = System.Drawing.SystemColors.Window;
            this.groupBox3.Controls.Add(this.downWallSpecularCB);
            this.groupBox3.Controls.Add(this.upWallSpecularCB);
            this.groupBox3.Controls.Add(this.rightWallSpecularCB);
            this.groupBox3.Controls.Add(this.leftWallSpecularCB);
            this.groupBox3.Controls.Add(this.backWallSpecularCB);
            this.groupBox3.Controls.Add(this.frontWallSpecularCB);
            this.groupBox3.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.875F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.groupBox3.Location = new System.Drawing.Point(790, 16);
            this.groupBox3.Margin = new System.Windows.Forms.Padding(6);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Padding = new System.Windows.Forms.Padding(6);
            this.groupBox3.RightToLeft = System.Windows.Forms.RightToLeft.No;
            this.groupBox3.Size = new System.Drawing.Size(166, 353);
            this.groupBox3.TabIndex = 5;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "Зеркальность стен";
            this.groupBox3.Enter += new System.EventHandler(this.groupBox3_Enter);
            // 
            // groupBox4
            // 
            this.groupBox4.BackColor = System.Drawing.SystemColors.Window;
            this.groupBox4.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.groupBox4.Controls.Add(this.twoLightsCB);
            this.groupBox4.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.875F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.groupBox4.Location = new System.Drawing.Point(790, 381);
            this.groupBox4.Margin = new System.Windows.Forms.Padding(6);
            this.groupBox4.Name = "groupBox4";
            this.groupBox4.Padding = new System.Windows.Forms.Padding(6);
            this.groupBox4.Size = new System.Drawing.Size(332, 76);
            this.groupBox4.TabIndex = 4;
            this.groupBox4.TabStop = false;
            this.groupBox4.Text = "Источник света";
            this.groupBox4.Enter += new System.EventHandler(this.groupBox4_Enter);
            // 
            // statusLabel
            // 
            this.statusLabel.AutoSize = true;
            this.statusLabel.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.statusLabel.Location = new System.Drawing.Point(790, 668);
            this.statusLabel.Name = "statusLabel";
            this.statusLabel.Size = new System.Drawing.Size(144, 16);
            this.statusLabel.TabIndex = 6;
            this.statusLabel.Text = "Готово к рендерингу";
            this.statusLabel.Click += new System.EventHandler(this.statusLabel_Click);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.SystemColors.Control;
            this.ClientSize = new System.Drawing.Size(1229, 724);
            this.Controls.Add(this.statusLabel);
            this.Controls.Add(this.groupBox4);
            this.Controls.Add(this.groupBox3);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.pictureBox1);
            this.Font = new System.Drawing.Font("Symbol", 8.25F);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Fixed3D;
            this.Margin = new System.Windows.Forms.Padding(4);
            this.MaximizeBox = false;
            this.Name = "Form1";
            this.Text = "Ray Tracing";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.groupBox3.ResumeLayout(false);
            this.groupBox3.PerformLayout();
            this.groupBox4.ResumeLayout(false);
            this.groupBox4.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.CheckBox cubeSpecularCB;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.CheckBox sphereSpecularCB;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.CheckBox refractSphereCB;
        private System.Windows.Forms.CheckBox refractCubeCB;
        private System.Windows.Forms.CheckBox twoLightsCB;
        private System.Windows.Forms.CheckBox frontWallSpecularCB;
        private System.Windows.Forms.CheckBox backWallSpecularCB;
        private System.Windows.Forms.CheckBox leftWallSpecularCB;
        private System.Windows.Forms.CheckBox rightWallSpecularCB;
        private System.Windows.Forms.CheckBox upWallSpecularCB;
        private System.Windows.Forms.CheckBox downWallSpecularCB;
        private System.Windows.Forms.GroupBox groupBox3;
        private System.Windows.Forms.GroupBox groupBox4;
        private System.Windows.Forms.Label statusLabel;
    }
}

